"""
Distributed engine with full Megatron parallelism (MLP + attention heads).
Same structure as multi-process/src/engine.py but calls apply_full_tensor_parallelism.
"""

import os
import logging

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM

from parallel_full import apply_full_tensor_parallelism

logger = logging.getLogger(__name__)


class FullMegatronEngine:
    def __init__(
        self,
        model_name: str,
        rank: int,
        world_size: int = 2,
        max_batch_size: int = 8,
    ):
        self.model_name = model_name
        self.rank = rank
        self.world_size = world_size
        self.device = f"cuda:{rank}"

        self._load_tokenizer()
        self._load_model()

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=os.environ.get("HF_TOKEN"),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=os.environ.get("HF_TOKEN"),
            torch_dtype=torch.float16,
            device_map=self.device,
        )

        counts = apply_full_tensor_parallelism(model, self.rank, self.world_size)
        if self.rank == 0:
            print(f"Full Megatron applied: {counts['mlp']} MLP blocks + {counts['attn']} attention modules")

        model.eval()
        self.model = model

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 200,
        temperature: float = 0.0,
    ) -> list[str] | None:
        """
        Must be called on ALL ranks simultaneously.
        Returns decoded strings on rank 0, None on other ranks.
        temperature=0 (greedy) keeps all ranks in lockstep deterministically.
        """
        input_ids, attention_mask = self._broadcast_inputs(prompts)

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        if self.rank == 0:
            return self._decode(input_ids, output_ids)
        return None

    def _broadcast_inputs(self, prompts):
        if self.rank == 0:
            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            input_ids      = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            shape = torch.tensor(list(input_ids.shape), dtype=torch.long, device=self.device)
            dist.broadcast(shape, src=0)
            dist.broadcast(input_ids, src=0)
            dist.broadcast(attention_mask, src=0)
        else:
            shape = torch.zeros(2, dtype=torch.long, device=self.device)
            dist.broadcast(shape, src=0)
            B, S = shape.tolist()
            input_ids      = torch.zeros(B, S, dtype=torch.long, device=self.device)
            attention_mask = torch.zeros(B, S, dtype=torch.long, device=self.device)
            dist.broadcast(input_ids, src=0)
            dist.broadcast(attention_mask, src=0)

        return input_ids, attention_mask

    def _decode(self, input_ids, output_ids):
        prompt_len = input_ids.shape[1]
        results = []
        for i in range(output_ids.shape[0]):
            text = self.tokenizer.decode(
                output_ids[i, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            results.append(text.strip())
        return results

    @property
    def device_info(self):
        info = {}
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            info[f"cuda:{i}"] = {"allocated_gb": round(allocated, 2), "total_gb": round(total, 2)}
        return info
