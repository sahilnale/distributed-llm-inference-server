"""
Multi-process distributed inference benchmark.

Launch with torchrun (not python directly):
  torchrun --nproc_per_node=2 benchmarks/benchmark.py

torchrun spawns one process per GPU, sets RANK / LOCAL_RANK / WORLD_SIZE
environment variables, and initializes the NCCL process group. Each process
calls main() independently but stays in lockstep via collective operations.

Results written to benchmarks/results/multi_process.json (rank 0 only).
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from engine import DistributedEngine

logging.basicConfig(
    level=logging.WARNING,
)

MODEL_NAME = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
MAX_NEW_TOKENS = 200
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PROMPTS = [
    "Explain the transformer architecture in simple terms.",
    "What is the difference between supervised and unsupervised learning?",
    "How does backpropagation work in neural networks?",
    "What are the key advantages of attention mechanisms?",
    "Describe the concept of transfer learning and its applications.",
    "What is gradient descent and why is it used in machine learning?",
    "Explain how convolutional neural networks process images.",
    "What is the role of activation functions in deep learning?",
] * 13  # 104 prompts → 13 batches of 8


def log(msg: str, rank: int):
    if rank == 0:
        print(msg)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_throughput_experiment(engine: DistributedEngine, rank: int) -> dict:
    """100 requests at batch size 8, measure req/s and tok/s."""
    prompts = PROMPTS[:100]
    batches = [prompts[i:i+8] for i in range(0, len(prompts), 8)]

    log(f"\n[Experiment 1] Throughput — {len(prompts)} requests, {MAX_NEW_TOKENS} tokens each", rank)

    latencies = []
    for i, batch in enumerate(batches):
        t0 = time.perf_counter()
        engine.generate(batch, max_new_tokens=MAX_NEW_TOKENS)  # all ranks participate
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000
        if rank == 0:
            latencies.append(elapsed_ms)
            log(f"  Batch {i+1}/{len(batches)}: {len(batch)} requests in {elapsed_ms:.0f}ms", rank)

    if rank != 0:
        return {}

    per_req = [ms / 8 for ms in latencies[1:]]  # skip first (warm-up)
    per_req_sorted = sorted(per_req)
    p99 = per_req_sorted[int(len(per_req_sorted) * 0.99)]

    total_requests = sum(len(b) for b in batches)
    total_time = sum(latencies[1:]) / 1000
    req_per_sec = total_requests / total_time
    tok_per_sec = req_per_sec * MAX_NEW_TOKENS

    log(f"  → {req_per_sec:.2f} req/s, {tok_per_sec:.1f} tok/s, p99={p99:.1f}ms", rank)

    return {
        "requests_per_sec": round(req_per_sec, 3),
        "tokens_per_sec": round(tok_per_sec, 1),
        "avg_latency_ms": round(sum(per_req) / len(per_req), 1),
        "p99_latency_ms": round(p99, 1),
        "total_requests": total_requests,
    }


def run_concurrency_experiment(engine: DistributedEngine, rank: int) -> dict:
    """Vary batch size (simulated concurrency), measure scaling."""
    log(f"\n[Experiment 2] Concurrency scaling", rank)
    results = []

    for concurrency in [1, 2, 4, 8, 16]:
        batch = PROMPTS[:concurrency]
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            engine.generate(batch, max_new_tokens=MAX_NEW_TOKENS)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_s = sum(times[1:]) / len(times[1:])  # skip warm-up
        req_per_sec = concurrency / avg_s
        log(f"  concurrency={concurrency}: {req_per_sec:.2f} req/s, {avg_s*1000:.0f}ms latency", rank)

        if rank == 0:
            results.append({
                "concurrency": concurrency,
                "requests_per_sec": round(req_per_sec, 3),
                "avg_latency_ms": round(avg_s * 1000, 1),
            })

    return results


def run_batch_experiment(engine: DistributedEngine, rank: int) -> dict:
    """Vary batch size, measure throughput impact."""
    log(f"\n[Experiment 3] Batch size impact", rank)
    results = []

    for batch_size in [1, 2, 4, 8]:
        batch = PROMPTS[:batch_size]
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            engine.generate(batch, max_new_tokens=MAX_NEW_TOKENS)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_s = sum(times[1:]) / len(times[1:])
        req_per_sec = batch_size / avg_s
        log(f"  batch_size={batch_size}: {req_per_sec:.2f} req/s, {avg_s*1000:.0f}ms/batch", rank)

        if rank == 0:
            results.append({
                "batch_size": batch_size,
                "requests_per_sec": round(req_per_sec, 3),
                "avg_batch_latency_ms": round(avg_s * 1000, 1),
                "tokens_per_sec": round(req_per_sec * MAX_NEW_TOKENS, 1),
            })

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    log(f"\nFound {torch.cuda.device_count()} GPUs:", rank)
    if rank == 0:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  cuda:{i} — {props.name}, {props.total_memory / 1024**3:.1f}GB VRAM")

    log(f"\nLoading {MODEL_NAME} with NCCL tensor parallelism ({world_size} GPUs)...", rank)
    engine = DistributedEngine(MODEL_NAME, rank, world_size)

    if rank == 0:
        print(f"GPU memory after load: {engine.device_info}")

    # Warm-up (both ranks participate)
    log("\nWarm-up pass...", rank)
    engine.generate(PROMPTS[:2], max_new_tokens=10)

    # Experiments — all ranks participate, only rank 0 records timing
    throughput   = run_throughput_experiment(engine, rank)
    concurrency  = run_concurrency_experiment(engine, rank)
    batch_impact = run_batch_experiment(engine, rank)

    if rank == 0:
        results = {
            "config": {
                "model": MODEL_NAME,
                "num_gpus": world_size,
                "parallelism": "nccl_megatron_mlp",
                "max_new_tokens": MAX_NEW_TOKENS,
                "gpu_memory": engine.device_info,
                "note": "MLP tensor-parallel with NCCL all_reduce; attention replicated",
            },
            "throughput": throughput,
            "concurrency_scaling": concurrency,
            "batch_size_impact": batch_impact,
        }

        out_path = RESULTS_DIR / "multi_process.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {out_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
