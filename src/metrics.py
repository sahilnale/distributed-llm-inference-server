"""
Prometheus metrics for the inference server.

How Prometheus works with this app:
  - This module defines the metric objects (counters, histograms, gauges).
  - server.py exposes a /metrics HTTP endpoint that prometheus_client serves.
  - Prometheus scrapes that endpoint on a schedule (e.g., every 15s).
  - You call the helpers below (record_request, record_latency, etc.) from
    your request handler — the library aggregates values automatically.

Multi-process note:
  - Gunicorn/uvicorn spawn multiple worker processes. Each has its own memory.
  - PROMETHEUS_MULTIPROC_DIR must point to a shared temp dir so prometheus_client
    can aggregate across workers before serving /metrics.
  - Set that env var before importing this module (server.py handles it).
"""

import os
import time

import torch
from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, multiprocess

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# In multi-process mode we can't use the default global registry — we need
# one that knows to read from the shared temp dir. This registry is what
# server.py will hand to the /metrics endpoint.

def make_registry() -> CollectorRegistry:
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return registry


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------
# Each metric has a name (what Prometheus stores), a help string (human
# description), and optional label names (dimensions you can filter by).
#
# Labels let you slice one metric multiple ways. gpu_memory_used_bytes has a
# "device" label so you get one time-series per GPU instead of one total.

REQUESTS_TOTAL = Counter(
    "requests_total",
    "Total number of generate requests received",
    labelnames=["status"],   # "success" or "error"
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "End-to-end latency per request",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

TOKENS_PER_SECOND = Histogram(
    "tokens_per_second",
    "Tokens generated per second for each request",
    buckets=[10, 25, 50, 100, 200, 400, 800, 1600],
)

BATCH_SIZE = Histogram(
    "batch_size",
    "Number of requests processed together in one inference pass",
    buckets=[1, 2, 4, 8, 16],
)

QUEUE_DEPTH = Gauge(
    "queue_depth",
    "Current number of requests waiting in the Redis queue",
)

GPU_MEMORY_USED = Gauge(
    "gpu_memory_used_bytes",
    "GPU VRAM currently allocated (bytes)",
    labelnames=["device"],
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
# Callers import these instead of touching the metric objects directly.
# That keeps the instrumentation calls readable in business logic.

def record_request(status: str) -> None:
    """Increment the request counter. status should be 'success' or 'error'."""
    REQUESTS_TOTAL.labels(status=status).inc()


def record_latency(seconds: float) -> None:
    """Record end-to-end request latency."""
    REQUEST_LATENCY.observe(seconds)


def record_tokens_per_second(tps: float) -> None:
    """Record throughput for a single completed request."""
    TOKENS_PER_SECOND.observe(tps)


def record_batch(size: int) -> None:
    """Record how many requests were batched in one inference pass."""
    BATCH_SIZE.observe(size)


def set_queue_depth(depth: int) -> None:
    """Set the current queue depth (called by scheduler after each enqueue/dequeue)."""
    QUEUE_DEPTH.set(depth)


def update_gpu_memory() -> None:
    """
    Sample current VRAM usage for all visible GPUs and update gauges.

    torch.cuda.memory_allocated() returns bytes currently held by tensors.
    It does NOT include fragmentation overhead — torch.cuda.memory_reserved()
    is the full allocation from the driver. We track allocated (what the model
    actually uses) because reserved fluctuates with the allocator's caching.
    """
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i)
        GPU_MEMORY_USED.labels(device=f"cuda:{i}").set(allocated)


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------
# Usage:
#   with LatencyTimer() as t:
#       result = do_inference(...)
#   record_latency(t.elapsed)
#
# This pattern keeps timing concerns out of the core logic.

class LatencyTimer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start
