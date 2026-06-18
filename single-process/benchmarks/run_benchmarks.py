"""
Master benchmark runner — runs both single and multi GPU experiments,
computes speedups, and saves a combined results file.

Output: benchmarks/results/comparison.json

Usage:
  # Run everything
  python benchmarks/run_benchmarks.py

  # Skip single GPU (if already have results)
  python benchmarks/run_benchmarks.py --skip-single

  # Skip multi GPU (only 1 GPU available)
  python benchmarks/run_benchmarks.py --skip-multi
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

RESULTS_DIR = Path(__file__).parent / "results"


def compute_speedup(single: dict, multi: dict) -> dict:
    """
    Compare single vs multi GPU results and compute speedup ratios.

    Speedup > 1.0 means multi-GPU is faster.
    Perfect linear scaling would be 2.0 — we expect 1.5-1.9x due to
    communication overhead from the all-reduce between layers.
    """
    s = single["throughput"]
    m = multi["throughput"]

    return {
        "experiment": "single_vs_multi_gpu",
        "single_gpu": {
            "requests_per_second": s["requests_per_second"],
            "avg_latency_ms": s["avg_latency_ms"],
            "p99_latency_ms": s["p99_latency_ms"],
            "tokens_per_second": s["tokens_per_second"],
        },
        "multi_gpu": {
            "requests_per_second": m["requests_per_second"],
            "avg_latency_ms": m["avg_latency_ms"],
            "p99_latency_ms": m["p99_latency_ms"],
            "tokens_per_second": m["tokens_per_second"],
        },
        "speedup": {
            # How many times faster is multi-GPU?
            "throughput": round(m["requests_per_second"] / s["requests_per_second"], 2),
            "latency":    round(s["avg_latency_ms"] / m["avg_latency_ms"], 2),
            "tokens_per_second": round(m["tokens_per_second"] / s["tokens_per_second"], 2),
        },
        "efficiency": {
            # Perfect 2-GPU scaling = 2.0x. Efficiency = actual / perfect.
            # e.g. 1.7x speedup on 2 GPUs = 85% efficiency
            "scaling_efficiency_pct": round(
                (m["requests_per_second"] / s["requests_per_second"]) / 2.0 * 100, 1
            ),
        },
    }


def compare_concurrency(single: dict, multi: dict) -> dict:
    """Compare concurrency scaling curves between single and multi GPU."""
    result = {}
    for level in single["concurrency_scaling"]:
        if str(level) not in multi["concurrency_scaling"] and level not in multi["concurrency_scaling"]:
            continue
        s = single["concurrency_scaling"][level]
        m = multi["concurrency_scaling"].get(level, multi["concurrency_scaling"].get(str(level), {}))
        if not m:
            continue
        result[level] = {
            "single_gpu_rps": s["requests_per_second"],
            "multi_gpu_rps": m["requests_per_second"],
            "speedup": round(m["requests_per_second"] / s["requests_per_second"], 2),
        }
    return result


def print_summary(comparison: dict):
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)

    s = comparison["throughput"]["single_gpu"]
    m = comparison["throughput"]["multi_gpu"]
    sp = comparison["throughput"]["speedup"]

    print(f"\nThroughput (100 requests, {200} tokens each):")
    print(f"  Single GPU:  {s['requests_per_second']} req/s  |  {s['tokens_per_second']} tok/s  |  p99={s['p99_latency_ms']}ms")
    print(f"  Multi GPU:   {m['requests_per_second']} req/s  |  {m['tokens_per_second']} tok/s  |  p99={m['p99_latency_ms']}ms")
    print(f"  Speedup:     {sp['throughput']}x throughput  |  {sp['latency']}x latency  |  {sp['tokens_per_second']}x tok/s")
    print(f"  Efficiency:  {comparison['throughput']['efficiency']['scaling_efficiency_pct']}% of linear scaling")

    print(f"\nConcurrency scaling speedups:")
    for level, data in comparison["concurrency_scaling"].items():
        print(f"  concurrency={level}: {data['speedup']}x  ({data['single_gpu_rps']} → {data['multi_gpu_rps']} req/s)")

    print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-single", action="store_true", help="Use existing single GPU results")
    parser.add_argument("--skip-multi", action="store_true", help="Use existing multi GPU results")
    args = parser.parse_args()

    # Run or load single GPU results
    if args.skip_single:
        path = RESULTS_DIR / "single_gpu.json"
        if not path.exists():
            print("ERROR: single_gpu.json not found. Run without --skip-single first.")
            sys.exit(1)
        single_results = json.loads(path.read_text())
        print("Loaded existing single GPU results.")
    else:
        from single_gpu import main as run_single
        single_results = run_single()

    # Run or load multi GPU results
    if args.skip_multi:
        path = RESULTS_DIR / "multi_gpu.json"
        if not path.exists():
            print("ERROR: multi_gpu.json not found. Run without --skip-multi first.")
            sys.exit(1)
        multi_results = json.loads(path.read_text())
        print("Loaded existing multi GPU results.")
    else:
        from multi_gpu import main as run_multi
        multi_results = run_multi()

    # Build comparison
    comparison = {
        "throughput": compute_speedup(single_results, multi_results),
        "concurrency_scaling": compare_concurrency(single_results, multi_results),
        "configs": {
            "single_gpu": single_results["config"],
            "multi_gpu": multi_results["config"],
        },
    }

    out_path = RESULTS_DIR / "comparison.json"
    out_path.write_text(json.dumps(comparison, indent=2))
    print(f"\nComparison saved to {out_path}")

    print_summary(comparison)


if __name__ == "__main__":
    main()
