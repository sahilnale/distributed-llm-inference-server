"""
Inference engine — owns the model object and exposes generate().

Lifecycle:
  1. InferenceEngine.__init__() loads the tokenizer and model weights into GPU
     memory. This happens once at server startup, not per request.
  2. engine.generate(prompts, max_tokens, temperature) is called per batch.
     It tokenizes, runs model.generate(), decodes, and returns strings.

Single vs multi-GPU:
  - NUM_GPUS=1: model loads entirely onto cuda:0.
  - NUM_GPUS=2: parallel.py shards weight matrices across cuda:0 and cuda:1.
    After that, generate() is identical — parallelism is transparent.
"""

import os
import logging
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import metrics

logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(
        self,
        model_name: str,
        num_gpus: int = 1,
        max_batch_size: int = 8,
    ):
        self.model_name = model_name
        self.num_gpus = num_gpus
        self.max_batch_size = max_batch_size

        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPUs found. This server requires at least one GPU.")

        logger.info(f"Loading tokenizer: {model_name}")
        self._load_tokenizer()

        logger.info(f"Loading model on {num_gpus} GPU(s)")
        self._load_model()

        metrics.update_gpu_memory()
        logger.info("Engine ready.")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=os.environ.get("HF_TOKEN"),
        )
        # Llama's tokenizer has no pad token by default — it only ever
        # trained on single sequences. We set it to eos so the model
        # treats padding positions the same as end-of-sequence.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Pad on the left so that the last real token of every sequence
        # in the batch is at the same position. This matters for causal
        # (left-to-right) models: the generation starts from the rightmost
        # token, so all prompts must be right-aligned.
        self.tokenizer.padding_side = "left"

    def _load_model(self):
        if self.num_gpus == 1:
            self.model = self._load_single_gpu()
        else:
            self.model = self._load_multi_gpu()

        self.model.eval()  # disable dropout, put in inference mode

    def _load_single_gpu(self) -> AutoModelForCausalLM:
        return AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=os.environ.get("HF_TOKEN"),
            torch_dtype=torch.float16,   # fp16: half the VRAM of fp32
            device_map="cuda:0",         # entire model on GPU 0
        )

    def _load_multi_gpu(self) -> AutoModelForCausalLM:
        # Import here so single-GPU mode doesn't need torch.distributed
        from parallel import apply_tensor_parallelism

        # Load onto CPU first — parallel.py will move shards to each GPU
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=os.environ.get("HF_TOKEN"),
            torch_dtype=torch.float16,
            device_map="cpu",
        )

        # apply_tensor_parallelism modifies model in-place:
        # splits Linear weight matrices across cuda:0 and cuda:1,
        # and patches each layer's forward() to do the all-reduce.
        apply_tensor_parallelism(model, devices=["cuda:0", "cuda:1"])
        return model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ) -> list[str]:
        """
        Generate completions for a batch of prompts.

        Returns a list of strings (one per prompt) containing only the
        newly generated text, not the original prompt.

        torch.inference_mode() is a stricter version of torch.no_grad().
        It disables gradient tracking and some autograd bookkeeping we
        don't need for inference, saving memory and time.
        """
        if not prompts:
            return []

        input_ids, attention_mask = self._tokenize(prompts)

        # temperature=0 means greedy (deterministic): always pick the
        # highest-probability token. Any value > 0 enables sampling.
        do_sample = temperature > 0

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            # Stop at eos so we don't pad output sequences
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return self._decode(input_ids, output_ids)

    def _tokenize(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert text prompts to padded token ID tensors on the correct device.

        Returns:
          input_ids:      (batch_size, seq_len) int64 tensor
          attention_mask: (batch_size, seq_len) int64 tensor — 1 for real
                          tokens, 0 for padding
        """
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",   # return PyTorch tensors
            padding=True,          # pad shorter sequences to match longest
            truncation=True,       # truncate if prompt exceeds model max length
            max_length=2048,
        )

        # Move tensors to the device where the first model layer lives.
        # For single GPU this is cuda:0. For tensor parallel the embedding
        # layer is on cuda:0, so this is always correct.
        device = next(self.model.parameters()).device
        return (
            encoded["input_ids"].to(device),
            encoded["attention_mask"].to(device),
        )

    def _decode(
        self,
        input_ids: torch.Tensor,
        output_ids: torch.Tensor,
    ) -> list[str]:
        """
        Decode only the newly generated tokens (strip the prompt prefix).

        output_ids contains the full sequence: [prompt tokens] + [new tokens].
        We slice off the prompt portion so callers only see the completion.
        """
        prompt_len = input_ids.shape[1]
        completions = []

        for i in range(output_ids.shape[0]):
            new_tokens = output_ids[i, prompt_len:]
            text = self.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,   # strip <eos>, <pad>, etc.
                clean_up_tokenization_spaces=True,
            )
            completions.append(text.strip())

        return completions

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def device_info(self) -> dict:
        """Return GPU memory stats for all visible devices."""
        info = {}
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            info[f"cuda:{i}"] = {
                "allocated_gb": round(allocated, 2),
                "total_gb": round(total, 2),
            }
        return info
