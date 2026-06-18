"""
Single GPU baseline benchmark.

Loads the model entirely on cuda:0 and runs three experiments:
  1. Throughput vs multi-GPU (100 requests, fixed concurrency)
  2. Scaling under concurrent load (ramp concurrency 1→16)
  3. Batch window impact (fixed concurrency, vary batch window)

Results are written to benchmarks/results/single_gpu.json.

Run directly:
  python benchmarks/single_gpu.py
"""

import json
import os
import sys
import time
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from engine import InferenceEngine

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
NUM_REQUESTS = 100
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7

TEST_PROMPTS = [
    "Explain transformer attention in one paragraph.",
    "What is the difference between fp16 and fp32?",
    "Describe how GPUs accelerate matrix multiplication.",
    "What is tensor parallelism in distributed inference?",
    "Explain the difference between latency and throughput.",
    "How does the softmax function work in attention?",
    "What is a key-value cache in LLM inference?",
    "Describe the role of layer normalization in transformers.",
]


def make_engine(num_gpus: int = 1) -> InferenceEngine:
    print(f"Loading model ({MODEL_NAME}) on {num_gpus} GPU(s)...")
    engine = InferenceEngine(model_name=MODEL_NAME, num_gpus=num_gpus)
    print(f"Model loaded. GPU memory: {engine.device_info}")
    return engine


def run_throughput_experiment(engine: InferenceEngine) -> dict:
    """
    Experiment 1: Raw throughput.

    Send NUM_REQUESTS prompts in batches of MAX_BATCH_SIZE.
    Measure total time, compute requests/sec and tokens/sec.

    We batch manually here (no scheduler) to isolate GPU performance
    from queue/batching overhead.
    """
    print(f"\n[Experiment 1] Throughput — {NUM_REQUESTS} requests, {MAX_NEW_TOKENS} tokens each")

    batch_size = 8
    prompts = [TEST_PROMPTS[i % len(TEST_PROMPTS)] for i in range(NUM_REQUESTS)]
    batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]

    latencies = []
    total_tokens = 0
    start_total = time.perf_counter()

    for i, batch in enumerate(batches):
        t0 = time.perf_counter()
        outputs = engine.generate(batch, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
        t1 = time.perf_counter()

        latencies.append(t1 - t0)
        total_tokens += sum(len(o.split()) for o in outputs)

        print(f"  Batch {i+1}/{len(batches)}: {len(batch)} requests in {(t1-t0)*1000:.0f}ms")

    total_time = time.perf_counter() - start_total

    latencies_ms = [l * 1000 / batch_size for l in latencies]  # per-request latency
    latencies_ms.sort()
    p50 = latencies_ms[len(latencies_ms) // 2]
    p99 = latencies_ms[int(len(latencies_ms) * 0.99)]

    result = {
        "requests_per_second": round(NUM_REQUESTS / total_time, 2),
        "avg_latency_ms": round(sum(latencies_ms) / len(latencies_ms), 1),
        "p50_latency_ms": round(p50, 1),
        "p99_latency_ms": round(p99, 1),
        "tokens_per_second": round(total_tokens / total_time, 1),
        "total_time_s": round(total_time, 2),
    }
    print(f"  → {result['requests_per_second']} req/s, {result['tokens_per_second']} tok/s, p99={result['p99_latency_ms']}ms")
    return result


def run_concurrency_experiment(engine: InferenceEngine) -> dict:
    """
    Experiment 2: Scaling under concurrent load.

    Vary batch size (simulating concurrency levels) and measure how
    throughput and latency change. Shows where the GPU saturates.

    True concurrency requires async — here we approximate by varying
    batch size, which gives the same GPU utilization picture.
    """
    print(f"\n[Experiment 2] Concurrency scaling")

    concurrency_levels = [1, 2, 4, 8, 16]
    results = {}

    for concurrency in concurrency_levels:
        # Use concurrency as batch size — equivalent GPU load
        prompts = [TEST_PROMPTS[i % len(TEST_PROMPTS)] for i in range(concurrency)]

        # Warm up
        engine.generate(prompts[:1], max_new_tokens=50, temperature=TEMPERATURE)

        # Measure over multiple runs for stability
        runs = 5
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            engine.generate(prompts, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
            times.append(time.perf_counter() - t0)

        avg_time = sum(times) / len(times)
        rps = concurrency / avg_time
        tps = concurrency * MAX_NEW_TOKENS / avg_time

        results[concurrency] = {
            "requests_per_second": round(rps, 2),
            "avg_latency_ms": round(avg_time * 1000, 1),
            "tokens_per_second": round(tps, 1),
        }
        print(f"  concurrency={concurrency}: {rps:.1f} req/s, {avg_time*1000:.0f}ms latency")

    return results


def run_batch_window_experiment(engine: InferenceEngine) -> dict:
    """
    Experiment 3: Batch size impact on latency vs throughput.

    Fixed number of requests, vary batch size. Shows the tradeoff:
    larger batches = higher throughput but higher per-request latency.
    """
    print(f"\n[Experiment 3] Batch size impact")

    batch_sizes = [1, 2, 4, 8]
    results = {}
    num_requests = 32  # fixed total

    for batch_size in batch_sizes:
        prompts = [TEST_PROMPTS[i % len(TEST_PROMPTS)] for i in range(num_requests)]
        batches = [prompts[i:i+batch_size] for i in range(0, num_requests, batch_size)]

        t0 = time.perf_counter()
        for batch in batches:
            engine.generate(batch, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
        total_time = time.perf_counter() - t0

        rps = num_requests / total_time
        avg_batch_latency = total_time / len(batches) * 1000

        results[batch_size] = {
            "requests_per_second": round(rps, 2),
            "avg_batch_latency_ms": round(avg_batch_latency, 1),
            "tokens_per_second": round(num_requests * MAX_NEW_TOKENS / total_time, 1),
        }
        print(f"  batch_size={batch_size}: {rps:.1f} req/s, {avg_batch_latency:.0f}ms/batch")

    return results


def main():
    engine = make_engine(num_gpus=1)

    results = {
        "config": {
            "model": MODEL_NAME,
            "num_gpus": 1,
            "max_new_tokens": MAX_NEW_TOKENS,
            "gpu_memory": engine.device_info,
        },
        "throughput": run_throughput_experiment(engine),
        "concurrency_scaling": run_concurrency_experiment(engine),
        "batch_size_impact": run_batch_window_experiment(engine),
    }

    out_path = RESULTS_DIR / "single_gpu.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    main()
