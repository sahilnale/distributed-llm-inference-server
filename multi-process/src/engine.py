"""
Distributed inference engine — one instance per GPU rank.

Each rank:
  1. Loads the full model onto its GPU (from_pretrained with device_map=cuda:rank)
  2. apply_dist_tensor_parallelism() replaces MLP blocks in-place — each rank
     discards the weight slices it doesn't own, keeping only its shard.
  3. generate() broadcasts the tokenized input from rank 0 to all ranks, runs
     model.generate() on every rank in lockstep (NCCL all_reduce fires inside
     each MLP forward), then returns the decoded result only on rank 0.

Why lockstep is required:
  dist.all_reduce() is a collective operation — every rank must call it at the
  same point in the forward pass. If rank 0 calls model.generate() and rank 1
  doesn't, the all_reduce on rank 0 will hang waiting for rank 1.

  The pattern: both ranks call engine.generate() for every batch. Rank 1 runs
  the full forward pass (contributing its weight shards to the all_reduce) but
  discards its decoded output. Only rank 0 returns the result.
"""

import os
import logging

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM

from parallel_dist import apply_dist_tensor_parallelism

logger = logging.getLogger(__name__)


class DistributedEngine:
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
        self.max_batch_size = max_batch_size
        self.device = f"cuda:{rank}"

        logger.info(f"[rank {rank}] Loading tokenizer")
        self._load_tokenizer()

        logger.info(f"[rank {rank}] Loading model onto {self.device}")
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
        # Each rank loads the full model onto its GPU first.
        # apply_dist_tensor_parallelism then replaces MLP blocks with sharded
        # versions, freeing the weight slices this rank doesn't own.
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=os.environ.get("HF_TOKEN"),
            torch_dtype=torch.float16,
            device_map=self.device,
        )

        n = apply_dist_tensor_parallelism(model, self.rank, self.world_size)
        logger.info(f"[rank {self.rank}] Replaced {n} MLP blocks with NCCL-backed shards")

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
        Generate completions for a batch of prompts.

        Must be called on ALL ranks simultaneously — the forward pass
        uses NCCL collectives that block until every rank participates.

        temperature=0 (default) uses greedy decoding. This avoids a
        correctness issue: if sampling is stochastic, rank 0 and rank 1
        could draw different next tokens, causing their inputs to diverge
        on subsequent steps. Greedy is deterministic so all ranks stay
        in lockstep without explicit token broadcast.

        Returns decoded strings on rank 0, None on all other ranks.
        """
        input_ids, attention_mask = self._broadcast_inputs(prompts)

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,      # greedy — deterministic across ranks
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        if self.rank == 0:
            return self._decode(input_ids, output_ids)
        return None

    def _broadcast_inputs(
        self, prompts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize on rank 0 and broadcast to all other ranks.

        NCCL broadcast requires the tensor to already exist on all ranks
        (same shape). We send the shape first as a 2-element tensor, then
        allocate and fill on receiving ranks before broadcasting the data.
        """
        if self.rank == 0:
            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            input_ids     = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            # send shape so other ranks can allocate
            shape = torch.tensor(
                list(input_ids.shape), dtype=torch.long, device=self.device
            )
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

    def _decode(
        self, input_ids: torch.Tensor, output_ids: torch.Tensor
    ) -> list[str]:
        prompt_len = input_ids.shape[1]
        results = []
        for i in range(output_ids.shape[0]):
            new_tokens = output_ids[i, prompt_len:]
            text = self.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            results.append(text.strip())
        return results

    @property
    def device_info(self) -> dict:
        info = {}
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            info[f"cuda:{i}"] = {
                "allocated_gb": round(allocated, 2),
                "total_gb": round(total, 2),
            }
        return info
