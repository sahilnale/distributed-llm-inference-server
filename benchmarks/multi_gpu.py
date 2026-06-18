"""
Multi-GPU benchmark — same experiments as single_gpu.py but with tensor parallelism.

Loads the model sharded across cuda:0 and cuda:1 via parallel.py,
then runs identical experiments so results are directly comparable.

Results written to benchmarks/results/multi_gpu.json.
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


def main():
    if torch.cuda.device_count() < 2:
        print("ERROR: Multi-GPU benchmark requires at least 2 GPUs.")
        print(f"Found: {torch.cuda.device_count()} GPU(s)")
        sys.exit(1)

    print(f"Found {torch.cuda.device_count()} GPUs:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} — {props.name}, {props.total_memory / 1024**3:.1f}GB VRAM")

    print(f"\nLoading model ({MODEL_NAME}) with tensor parallelism across 2 GPUs...")
    engine = InferenceEngine(model_name=MODEL_NAME, num_gpus=2)
    print(f"Model loaded. GPU memory: {engine.device_info}")

    results = {
        "config": {
            "model": MODEL_NAME,
            "num_gpus": 2,
            "max_new_tokens": MAX_NEW_TOKENS,
            "gpu_memory": engine.device_info,
        },
        "throughput": run_throughput_experiment(engine),
        "concurrency_scaling": run_concurrency_experiment(engine),
        "batch_size_impact": run_batch_window_experiment(engine),
    }

    out_path = RESULTS_DIR / "multi_gpu.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    main()
