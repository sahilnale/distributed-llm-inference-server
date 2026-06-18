"""
Megatron-style tensor parallelism.

Reference: Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter
Language Models Using Model Parallelism" (2019). arXiv:1909.08053
https://arxiv.org/abs/1909.08053

Key improvement over parallel.py:
  parallel.py (naive column-parallel):
    Every Linear layer: copy input to GPU1, compute, gather output back.
    Cost: 2 cross-GPU transfers × 225 layers = 450 transfers per forward pass.

  parallel_megatron.py (alternating col/row):
    MLP block: col-parallel gate/up → (no transfer) → row-parallel down → all-reduce.
    Cost: 2 transfers per MLP block × 32 layers = 64 MLP transfers.
    Attention: col-parallel q/k/v → gather → attention → col-parallel o.
    Cost: ~5 transfers per attention block × 32 layers = 160 attention transfers.
    Total: ~224 transfers vs 450 — roughly 2x fewer syncs.

The core insight (from the paper):
  A column-parallel layer produces output split across GPUs.
  A row-parallel layer expects input split across GPUs.
  Back-to-back col→row layers require ZERO communication in the middle —
  the split output of col feeds directly into the split input of row.
  Only one all-reduce (sum) is needed at the end of the pair.

What this file implements:
  MegatronMLP   — true col/row alternation for the MLP block (gate/up/down)
  apply_megatron_tensor_parallelism() — replaces MLP blocks in-place;
                  attention uses col-parallel from parallel.py (simpler,
                  avoids reimplementing rotary embeddings and KV cache).

What full Megatron does additionally:
  - Splits attention heads across GPUs (16 heads per GPU for a 32-head model)
  - Each GPU runs attention independently on its heads, then all-reduce on o_proj
  - Requires patching the entire attention forward including RoPE and KV cache
  - Reduces attention transfers from ~160 to ~64 (one all-reduce per block)
"""

import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from parallel import ColumnParallelLinear, _is_shardable, _SHARDABLE_SUFFIXES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MegatronMLP
# ---------------------------------------------------------------------------

class MegatronMLP(nn.Module):
    """
    Replaces a transformer MLP block (gate_proj, up_proj, down_proj) with
    true Megatron-style tensor parallelism.

    SwiGLU MLP forward (what HuggingFace Mistral/Llama does):
      h = act_fn(gate_proj(x)) * up_proj(x)
      output = down_proj(h)

    Parallelism strategy:
      gate_proj, up_proj: column-parallel (split output features, NO gather)
      down_proj:          row-parallel    (take split input, sum via all-reduce)

    Data flow:
      x (full, GPU0) ──copy──→ x (full, GPU1)
           │                        │
      gate_0 = x @ G0.T       gate_1 = x @ G1.T     [column parallel, no gather]
      up_0   = x @ U0.T       up_1   = x @ U1.T     [column parallel, no gather]
           │                        │
      h_0 = act(gate_0)*up_0   h_1 = act(gate_1)*up_1   [element-wise, no comm]
           │                        │
      out_0 = h_0 @ D0.T       out_1 = h_1 @ D1.T   [row parallel]
           │                        │
           └──── out_0 + out_1.to(GPU0) ────┘         [all-reduce = sum]

    Cross-GPU transfers: 2 per MLP block (copy x, sum output)
    vs naive col-parallel: 6 per MLP block (2 per linear × 3 linears)
    """

    def __init__(self, mlp: nn.Module, devices: List[str]):
        super().__init__()
        d0, d1 = devices
        self.d0 = d0
        self.d1 = d1

        gate_W = mlp.gate_proj.weight.data
        up_W   = mlp.up_proj.weight.data
        down_W = mlp.down_proj.weight.data

        half_out = gate_W.shape[0] // 2   # split output dim for col-parallel
        half_in  = down_W.shape[1] // 2   # split input dim for row-parallel

        # Column parallel: gate and up — split along output features
        self.gate_w0 = nn.Parameter(gate_W[:half_out].clone().to(d0), requires_grad=False)
        self.gate_w1 = nn.Parameter(gate_W[half_out:].clone().to(d1), requires_grad=False)

        self.up_w0 = nn.Parameter(up_W[:half_out].clone().to(d0), requires_grad=False)
        self.up_w1 = nn.Parameter(up_W[half_out:].clone().to(d1), requires_grad=False)

        # Row parallel: down — split along INPUT features (columns of W, rows of W^T)
        # GPU0 handles first half_in input features, GPU1 handles second half_in.
        # Each GPU computes a PARTIAL output (same shape as full output).
        # Summing the two partials gives the correct full output.
        self.down_w0 = nn.Parameter(down_W[:, :half_in].clone().to(d0), requires_grad=False)
        self.down_w1 = nn.Parameter(down_W[:, half_in:].clone().to(d1), requires_grad=False)

        self.act_fn = mlp.act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x arrives on d0 from the layer norm above
        x1 = x.to(self.d1)                               # 1 transfer: x → GPU1

        # Column parallel gate and up projections — outputs stay SPLIT
        gate_0 = F.linear(x,  self.gate_w0)              # (B, S, half_out) on GPU0
        gate_1 = F.linear(x1, self.gate_w1)              # (B, S, half_out) on GPU1

        up_0 = F.linear(x,  self.up_w0)                  # (B, S, half_out) on GPU0
        up_1 = F.linear(x1, self.up_w1)                  # (B, S, half_out) on GPU1

        # SwiGLU activation — element-wise, no communication needed
        h_0 = self.act_fn(gate_0) * up_0                 # on GPU0
        h_1 = self.act_fn(gate_1) * up_1                 # on GPU1

        # Row parallel down projection — each GPU computes partial output
        # h_0 is the left half of h, h_1 is the right half.
        # down_w0 handles left-half input → partial output on GPU0
        # down_w1 handles right-half input → partial output on GPU1
        out_0 = F.linear(h_0, self.down_w0)              # (B, S, hidden) partial on GPU0
        out_1 = F.linear(h_1, self.down_w1)              # (B, S, hidden) partial on GPU1

        # All-reduce: sum the two partial outputs           1 transfer: partial_1 → GPU0
        return out_0 + out_1.to(self.d0)


# ---------------------------------------------------------------------------
# Apply parallelism
# ---------------------------------------------------------------------------

def apply_megatron_tensor_parallelism(model: nn.Module, devices: List[str]) -> None:
    """
    Apply Megatron-style parallelism to the model in-place.

    Strategy:
      - MLP blocks → replaced with MegatronMLP (true col/row alternation)
      - Attention projections → replaced with ColumnParallelLinear from parallel.py
        (simpler; avoids reimplementing rotary embeddings and KV cache logic)

    The MLP replacement is the primary contribution — it accounts for ~40% of
    all parameters and reduces MLP communication from 6 transfers to 2 per block.
    """
    assert len(devices) == 2

    logger.info(f"Moving model to {devices[0]}")
    model.to(devices[0])

    mlp_count = 0
    attn_count = 0

    _replace_recursive(model, devices, "", mlp_count=[0], attn_count=[0])

    logger.info(f"Megatron parallelism applied: "
                f"MLP blocks replaced, attention col-parallel applied")


def _replace_recursive(
    model: nn.Module,
    devices: List[str],
    prefix: str,
    mlp_count: list,
    attn_count: list,
) -> None:
    for name, child in list(model.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        if _is_mlp_block(child):
            new_module = MegatronMLP(child, devices)
            setattr(model, name, new_module)
            mlp_count[0] += 1
            logger.debug(f"  MegatronMLP: {full_name}")

        elif isinstance(child, nn.Linear) and _is_shardable(full_name, child):
            # Attention projections: use existing col-parallel from parallel.py
            new_module = ColumnParallelLinear.from_linear(child, devices)
            setattr(model, name, new_module)
            attn_count[0] += 1
            logger.debug(f"  ColParallel: {full_name}")

        else:
            _replace_recursive(child, devices, full_name, mlp_count, attn_count)


def _is_mlp_block(module: nn.Module) -> bool:
    """
    Detect an MLP block by checking for the three projection attributes.
    Works for Llama, Mistral, and other SwiGLU MLP variants.
    """
    return (
        hasattr(module, "gate_proj") and
        hasattr(module, "up_proj") and
        hasattr(module, "down_proj") and
        isinstance(module.gate_proj, nn.Linear) and
        module.gate_proj.weight.shape[0] % 2 == 0
    )
