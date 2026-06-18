"""
Tensor parallelism — shard Linear layer weights across two GPUs.

Strategy: column parallelism on every qualifying Linear layer.
  - Weight W (out, in) is split into W_0 (out//2, in) and W_1 (out//2, in)
  - W_0 lives on cuda:0, W_1 lives on cuda:1
  - forward() runs both halves simultaneously, then concatenates on cuda:0

Production systems (Megatron-LM) alternate column-parallel and row-parallel
layers so tensors stay split between consecutive layers and only one
all-reduce is needed per transformer layer. We concat after every Linear
instead — more data movement, but the concept is identical and easier to
follow. See PRODUCTION_NOTE at the bottom for details.

Entry point:
  apply_tensor_parallelism(model, devices=["cuda:0", "cuda:1"])
  Called once by engine.py after loading the model onto CPU.
"""

import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# These name fragments identify the large projection matrices inside each
# transformer layer. We skip embeddings and layer norms — they're small
# and not linear projections.
_SHARDABLE_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention projections
    "gate_proj", "up_proj", "down_proj",        # MLP projections
    "lm_head",                                  # final vocab projection (~500MB)
)


# ---------------------------------------------------------------------------
# The replacement module
# ---------------------------------------------------------------------------

class ColumnParallelLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that splits computation across 2 GPUs.

    Column parallel means we split along the OUTPUT dimension (columns of W^T,
    or rows of W). Each GPU computes a subset of the output features.

    Shape walkthrough for Linear(4096, 4096):
      Original W:  (4096, 4096)
      W_0:         (2048, 4096)  on cuda:0
      W_1:         (2048, 4096)  on cuda:1

      input:       (batch, seq, 4096)  on cuda:0
      out_0:       (batch, seq, 2048)  on cuda:0  ← input @ W_0.T
      out_1:       (batch, seq, 2048)  on cuda:1  ← input @ W_1.T
      output:      (batch, seq, 4096)  on cuda:0  ← cat([out_0, out_1])
    """

    def __init__(
        self,
        weight_0: torch.Tensor,
        weight_1: torch.Tensor,
        bias_0: torch.Tensor | None,
        bias_1: torch.Tensor | None,
        device_0: str,
        device_1: str,
    ):
        super().__init__()
        # nn.Parameter tells PyTorch these are learnable tensors (even though
        # we won't train — it keeps them in .parameters() and .state_dict()).
        # requires_grad=False because we're inference-only.
        self.weight_0 = nn.Parameter(weight_0.to(device_0), requires_grad=False)
        self.weight_1 = nn.Parameter(weight_1.to(device_1), requires_grad=False)

        self.bias_0 = nn.Parameter(bias_0.to(device_0), requires_grad=False) if bias_0 is not None else None
        self.bias_1 = nn.Parameter(bias_1.to(device_1), requires_grad=False) if bias_1 is not None else None

        self.device_0 = device_0
        self.device_1 = device_1

    @classmethod
    def from_linear(cls, linear: nn.Linear, devices: List[str]) -> "ColumnParallelLinear":
        """
        Build a ColumnParallelLinear from an existing nn.Linear.

        We clone() each half so the original weight tensor can be freed
        from CPU memory after this call.
        """
        W = linear.weight.data          # (out_features, in_features)
        half = W.shape[0] // 2

        W_0 = W[:half].clone()          # first half of output features
        W_1 = W[half:].clone()          # second half of output features

        b_0 = b_1 = None
        if linear.bias is not None:
            b = linear.bias.data
            b_0 = b[:half].clone()
            b_1 = b[half:].clone()

        return cls(W_0, W_1, b_0, b_1, devices[0], devices[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x arrives on device_0 from the previous layer.

        We copy x to device_1 before running that half. The copy happens
        over PCIe (or NVLink if available) — this is the communication cost
        of tensor parallelism. On NVLink the bandwidth is ~600 GB/s and the
        copy is negligible. On PCIe (~32 GB/s) it starts to matter.

        Both F.linear calls can overlap in practice because CUDA operations
        on different devices are dispatched to different streams and execute
        concurrently. PyTorch handles this automatically.
        """
        # Run first half on cuda:0
        out_0 = F.linear(x, self.weight_0, self.bias_0)

        # Run second half on cuda:1 — copy input across, compute, copy result back
        out_1 = F.linear(x.to(self.device_1), self.weight_1, self.bias_1)

        # Gather: concatenate along the feature dimension on device_0
        # .to(self.device_0) blocks until device_1's computation is done,
        # effectively synchronizing the two streams here.
        return torch.cat([out_0, out_1.to(self.device_0)], dim=-1)


# ---------------------------------------------------------------------------
# Model surgery
# ---------------------------------------------------------------------------

def apply_tensor_parallelism(model: nn.Module, devices: List[str]) -> None:
    """
    Shard qualifying Linear layers across devices in-place.

    Called once at startup by engine.py. After this returns, model.generate()
    will use both GPUs automatically on every forward pass.

    Steps:
      1. Move entire model to devices[0] (all weights on cuda:0 initially).
      2. Walk the module tree.
      3. For each qualifying nn.Linear, replace it with ColumnParallelLinear
         (which moves half the weights to devices[1]).
    """
    assert len(devices) == 2, "Tensor parallelism currently supports exactly 2 GPUs"

    logger.info(f"Moving model to {devices[0]}")
    model.to(devices[0])

    replaced = 0
    _shard_recursive(model, devices, prefix="", count=[replaced])
    logger.info(f"Sharded {_shard_recursive.count} Linear layers across {devices}")


def _shard_recursive(model: nn.Module, devices: List[str], prefix: str, count: list) -> None:
    """
    Recursively walk model's children and replace qualifying Linear layers.

    We use named_children() (direct children only) and recurse manually
    rather than named_modules() (all descendants) so we can use setattr
    on the correct parent module.
    """
    for name, child in list(model.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        if _is_shardable(full_name, child):
            new_module = ColumnParallelLinear.from_linear(child, devices)
            setattr(model, name, new_module)
            count[0] += 1
            logger.debug(f"  Sharded {full_name}: {child.weight.shape}")
        else:
            # Recurse into non-replaced children
            _shard_recursive(child, devices, full_name, count)

# Attach count as a function attribute so apply_tensor_parallelism can log it
_shard_recursive.count = 0


def _is_shardable(name: str, module: nn.Module) -> bool:
    """
    Return True if this module should be replaced with ColumnParallelLinear.

    Conditions:
      - Must be an nn.Linear (has a weight matrix we can split)
      - Name must end with one of our target suffixes
      - Output dimension must be even (so we can split in half cleanly)
    """
    if not isinstance(module, nn.Linear):
        return False
    if not any(name.endswith(suffix) for suffix in _SHARDABLE_SUFFIXES):
        return False
    if module.weight.shape[0] % 2 != 0:
        return False
    return True


# ---------------------------------------------------------------------------
# PRODUCTION_NOTE
# ---------------------------------------------------------------------------
# Megatron-LM's column + row parallel alternation:
#
#   MLP block:
#     up_proj:   column parallel → output stays SPLIT across GPUs
#     down_proj: row parallel    → each GPU takes its split of the input,
#                                  computes partial output, then ALL-REDUCE (sum)
#
#   With this pattern the output of up_proj feeds directly into down_proj
#   without gathering first. Only one cross-GPU sync per MLP block instead
#   of one per Linear.
#
#   We gather (concat) after every Linear, doubling the syncs. For this
#   project the difference shows up in the benchmark — you'll see the
#   2-GPU speedup is real but less than 2x, partly due to this overhead.
#   That's an honest result worth explaining in the README.
