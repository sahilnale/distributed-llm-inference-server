"""
Full Megatron-LM tensor parallelism: MLP blocks + attention heads.

Reference: Shoeybi et al., "Megatron-LM" (2019). arXiv:1909.08053

Extends multi-process/ which parallelized MLP only (attention replicated).
This file also splits attention heads — each rank handles a subset of heads
and runs the full attention computation on just those heads independently.
One NCCL all_reduce on o_proj combines the partial outputs.

Why attention parallelism matters:
  multi-process/:     MLP parallel, attention REPLICATED (both GPUs same work)
  full-megatron/:     MLP parallel, attention SPLIT (each GPU different heads)

Attention head split for Mistral 7B (world_size=2):
  num_heads    = 32 → 16 per rank
  num_kv_heads = 8  →  4 per rank   (GQA ratio 4:1 preserved per rank)
  head_dim     = 128 (unchanged)

Communication per transformer block:
  Attention o_proj: 1 NCCL all_reduce
  MLP down_proj:    1 NCCL all_reduce
  Total:            2 per block × 32 blocks = 64 all_reduces
  (same count as multi-process/ but now attention compute is also halved)
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLP — identical to multi-process/src/parallel_dist.py
# ---------------------------------------------------------------------------

class MegatronMLPFull(nn.Module):
    """Megatron col/row MLP with NCCL all_reduce."""

    def __init__(self, mlp: nn.Module, rank: int, world_size: int = 2):
        super().__init__()
        self.rank = rank

        gate_W = mlp.gate_proj.weight.data
        up_W   = mlp.up_proj.weight.data
        down_W = mlp.down_proj.weight.data

        half_out = gate_W.shape[0] // world_size
        half_in  = down_W.shape[1] // world_size
        device   = gate_W.device

        self.gate_w = nn.Parameter(gate_W[rank*half_out:(rank+1)*half_out].clone().to(device), requires_grad=False)
        self.up_w   = nn.Parameter(up_W  [rank*half_out:(rank+1)*half_out].clone().to(device), requires_grad=False)
        self.down_w = nn.Parameter(down_W[:, rank*half_in:(rank+1)*half_in].clone().to(device), requires_grad=False)
        self.act_fn = mlp.act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.linear(x, self.gate_w)
        up   = F.linear(x, self.up_w)
        h    = self.act_fn(gate) * up
        out  = F.linear(h, self.down_w)
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out


# ---------------------------------------------------------------------------
# Attention — column-parallel q/k/v, row-parallel o_proj
# ---------------------------------------------------------------------------

class TensorParallelAttention(nn.Module):
    """
    Patches a MistralAttention (or LlamaAttention) module to split heads.

    Each rank holds:
      q_proj weight: W_q[rank*local_q_dim : (rank+1)*local_q_dim, :]
      k_proj weight: W_k[rank*local_kv_dim : (rank+1)*local_kv_dim, :]
      v_proj weight: W_v[rank*local_kv_dim : (rank+1)*local_kv_dim, :]
      o_proj weight: W_o[:, rank*local_q_dim : (rank+1)*local_q_dim]

    Forward (per rank):
      q = x @ q_w.T  → (B, S, local_heads * head_dim)   [col-parallel, no comm]
      k = x @ k_w.T  → (B, S, local_kv_heads * head_dim) [col-parallel]
      v = x @ v_w.T  → (B, S, local_kv_heads * head_dim) [col-parallel]
      Apply RoPE, GQA repeat_kv — works correctly on local heads
      attn_out → (B, S, local_heads * head_dim)
      out = attn_out @ o_w.T  → (B, S, hidden) partial  [row-parallel]
      dist.all_reduce(out)     → NCCL sum → full output

    The GQA ratio (num_heads / num_kv_heads = 4) is preserved per rank
    (16q : 4kv = 4:1), so repeat_kv still works identically.
    """

    def __init__(self, attn: nn.Module, rank: int, world_size: int = 2):
        super().__init__()
        self.rank = rank
        self.world_size = world_size

        # Halve the head counts for this rank
        self.num_heads          = attn.num_heads // world_size
        self.num_kv_heads       = attn.num_key_value_heads // world_size
        self.num_kv_groups      = self.num_heads // self.num_kv_heads  # unchanged ratio
        self.head_dim           = attn.head_dim
        self.hidden_size        = attn.hidden_size
        self.attention_dropout  = attn.attention_dropout
        self.layer_idx          = attn.layer_idx

        local_q_dim  = self.num_heads   * self.head_dim   # 16 * 128 = 2048
        local_kv_dim = self.num_kv_heads * self.head_dim  # 4  * 128 = 512

        device = attn.q_proj.weight.device

        # Column-parallel q/k/v: each rank takes its head slice
        self.q_w = nn.Parameter(attn.q_proj.weight.data[rank*local_q_dim  : (rank+1)*local_q_dim ].clone().to(device), requires_grad=False)
        self.k_w = nn.Parameter(attn.k_proj.weight.data[rank*local_kv_dim : (rank+1)*local_kv_dim].clone().to(device), requires_grad=False)
        self.v_w = nn.Parameter(attn.v_proj.weight.data[rank*local_kv_dim : (rank+1)*local_kv_dim].clone().to(device), requires_grad=False)

        # Row-parallel o_proj: each rank takes its input column slice
        self.o_w = nn.Parameter(attn.o_proj.weight.data[:, rank*local_q_dim : (rank+1)*local_q_dim].clone().to(device), requires_grad=False)
        # o_proj bias (if present) added once after all_reduce
        self.o_bias = attn.o_proj.bias

        # Keep rotary embedding from original module
        self.rotary_emb = attn.rotary_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, None]:
        from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, repeat_kv

        bsz, q_len, _ = hidden_states.size()

        # Column-parallel projections — each rank computes its head subset
        q = F.linear(hidden_states, self.q_w)  # (B, S, local_heads * head_dim)
        k = F.linear(hidden_states, self.k_w)  # (B, S, local_kv_heads * head_dim)
        v = F.linear(hidden_states, self.v_w)

        q = q.view(bsz, q_len, self.num_heads,   self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE — position-dependent, works identically on local heads
        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV cache
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

        # GQA: repeat k/v for each query group (ratio preserved per rank)
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        # Scaled dot-product attention on local heads — no communication
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, :k.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)

        # Row-parallel o_proj: each rank contributes its partial output
        out = F.linear(attn_out, self.o_w)
        dist.all_reduce(out, op=dist.ReduceOp.SUM)  # NCCL: sum partial outputs
        if self.o_bias is not None:
            out = out + self.o_bias

        return out, None, past_key_value


# ---------------------------------------------------------------------------
# Apply full parallelism
# ---------------------------------------------------------------------------

def apply_full_tensor_parallelism(model: nn.Module, rank: int, world_size: int = 2) -> dict:
    """
    Replace MLP blocks and attention modules in-place.
    Returns counts of replaced modules.
    """
    counts = {"mlp": 0, "attn": 0}
    _replace_recursive(model, rank, world_size, counts)
    return counts


def _replace_recursive(module: nn.Module, rank: int, world_size: int, counts: dict):
    for name, child in list(module.named_children()):
        if _is_mlp_block(child):
            setattr(module, name, MegatronMLPFull(child, rank, world_size))
            counts["mlp"] += 1
        elif _is_attention_block(child):
            setattr(module, name, TensorParallelAttention(child, rank, world_size))
            counts["attn"] += 1
        else:
            _replace_recursive(child, rank, world_size, counts)


def _is_mlp_block(module: nn.Module) -> bool:
    return (
        hasattr(module, "gate_proj") and
        hasattr(module, "up_proj") and
        hasattr(module, "down_proj") and
        isinstance(module.gate_proj, nn.Linear) and
        module.gate_proj.weight.shape[0] % 2 == 0
    )


def _is_attention_block(module: nn.Module) -> bool:
    return (
        hasattr(module, "q_proj") and
        hasattr(module, "k_proj") and
        hasattr(module, "v_proj") and
        hasattr(module, "o_proj") and
        hasattr(module, "num_heads") and
        hasattr(module, "num_key_value_heads") and
        isinstance(module.q_proj, nn.Linear) and
        module.num_heads % 2 == 0 and
        module.num_key_value_heads % 2 == 0
    )
