"""
Multi-GPU benchmark — same experiments as single_gpu.py but with tensor parallelism.

Runs two parallelism strategies back-to-back for direct comparison:
  column   — naive column-parallel (parallel.py), 450 transfers/pass
  megatron — Megatron-style col/row alternation (parallel_megatron.py), ~224 transfers/pass
             Reference: Shoeybi et al. (2019) arXiv:1909.08053

Results written to:
  benchmarks/results/multi_gpu_column.json
  benchmarks/results/multi_gpu_megatron.json
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from engine import InferenceEngine
from single_gpu import (
    run_throughput_experiment,
    run_concurrency_experiment,
    run_batch_window_experiment,
    MODEL_NAME,
    MAX_NEW_TOKENS,
    RESULTS_DIR,
)

import torch


def run_mode(mode: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Parallelism mode: {mode}")
    print(f"{'='*60}")
    print(f"Loading {MODEL_NAME} with parallelism_mode='{mode}'...")

    engine = InferenceEngine(model_name=MODEL_NAME, num_gpus=2, parallelism_mode=mode)
    print(f"Model loaded. GPU memory: {engine.device_info}")

    results = {
        "config": {
            "model": MODEL_NAME,
            "num_gpus": 2,
            "parallelism_mode": mode,
            "max_new_tokens": MAX_NEW_TOKENS,
            "gpu_memory": engine.device_info,
        },
        "throughput": run_throughput_experiment(engine),
        "concurrency_scaling": run_concurrency_experiment(engine),
        "batch_size_impact": run_batch_window_experiment(engine),
    }

    out_path = RESULTS_DIR / f"multi_gpu_{mode}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

    # Free GPU memory before loading next mode
    del engine
    torch.cuda.empty_cache()

    return results


def main():
    if torch.cuda.device_count() < 2:
        print("ERROR: Multi-GPU benchmark requires at least 2 GPUs.")
        print(f"Found: {torch.cuda.device_count()} GPU(s)")
        sys.exit(1)

    print(f"Found {torch.cuda.device_count()} GPUs:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} — {props.name}, {props.total_memory / 1024**3:.1f}GB VRAM")

    column_results  = run_mode("column")
    megatron_results = run_mode("megatron")

    # Quick comparison summary
    col_thr  = column_results["throughput"]
    meg_thr  = megatron_results["throughput"]
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'column':>10} {'megatron':>10}")
    print(f"{'-'*50}")
    print(f"{'req/s':<30} {col_thr['requests_per_sec']:>10.2f} {meg_thr['requests_per_sec']:>10.2f}")
    print(f"{'avg latency (ms)':<30} {col_thr['avg_latency_ms']:>10.1f} {meg_thr['avg_latency_ms']:>10.1f}")
    print(f"{'tokens/s':<30} {col_thr['tokens_per_sec']:>10.1f} {meg_thr['tokens_per_sec']:>10.1f}")

    improvement = (meg_thr["requests_per_sec"] / col_thr["requests_per_sec"] - 1) * 100
    print(f"\nMegatron vs column speedup: {improvement:+.1f}%")

    return {"column": column_results, "megatron": megatron_results}


if __name__ == "__main__":
    main()
