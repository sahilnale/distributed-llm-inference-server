# Full Megatron — MLP + Attention Head Parallelism

Full implementation of Megatron-LM tensor parallelism: both MLP blocks and
attention heads are split across GPUs via NCCL. This is what production
inference engines (vLLM, TGI) actually run.

Reference: Shoeybi et al., *Megatron-LM* (2019). [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)

---

## What's New vs `multi-process/`

`multi-process/` parallelized MLP only — attention was replicated (both GPUs
ran the same computation). That left ~35% of model compute unsplit.

This folder also parallelizes attention:

| Module | `multi-process/` | `full-megatron/` |
|--------|-----------------|-----------------|
| MLP gate/up/down | NCCL col/row ✓ | NCCL col/row ✓ |
| Attention q/k/v | replicated ✗ | col-parallel ✓ |
| Attention o_proj | replicated ✗ | NCCL row-parallel ✓ |
| Attention compute | both GPUs same | each GPU different heads |

---

## How Attention Parallelism Works

Mistral 7B has 32 query heads and 8 KV heads (GQA, 4:1 ratio), head_dim=128.

```
Single GPU:
  q_proj: (4096 → 4096)   all 32 heads
  k_proj: (4096 → 1024)   all  8 KV heads
  v_proj: (4096 → 1024)   all  8 KV heads
  → attention over all 32 heads
  o_proj: (4096 → 4096)

Rank 0:                         Rank 1:
  q_proj: (4096 → 2048)           q_proj: (4096 → 2048)
  heads 0–15                       heads 16–31
  k_proj: (4096 → 512)             k_proj: (4096 → 512)
  kv_heads 0–3                      kv_heads 4–7
  → attention over 16 heads        → attention over 16 heads
  o_proj: partial (4096→4096)      o_proj: partial (4096→4096)
           │                                │
           └──── dist.all_reduce() ─────────┘
                 NCCL sum → full output
```

The GQA ratio (4:1) is preserved per rank (16q : 4kv = 4:1), so
`repeat_kv` and the attention math work identically without modification.

---

## How to Run

```bash
cd /workspace/distributed-llm-inference-server
torchrun --nproc_per_node=2 full-megatron/benchmarks/benchmark.py
```

---

## Benchmark Results

*Run on 2× V100 SXM2 (32GB, NVLink 154.7 GB/s) on Vast.ai*

| Metric | Single GPU | NCCL MLP-only | **Full Megatron (this)** |
|--------|-----------|---------------|--------------------------|
| Requests/sec | 1.10 | 1.04 | TBD |
| vs single GPU | 1.0x | 0.95x | TBD |

---

## Project Structure

```
full-megatron/
├── src/
│   ├── parallel_full.py   MegatronMLPFull + TensorParallelAttention
│   └── engine.py          FullMegatronEngine
├── benchmarks/
│   ├── benchmark.py       torchrun entry point
│   └── results/
└── README.md
```
