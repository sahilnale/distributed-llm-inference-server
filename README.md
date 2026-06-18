# Distributed LLM Inference Server

> Built a distributed LLM inference server implementing four tensor parallelism strategies on Mistral-7B — progressing from a 0.64x regression to a 1.05x speedup over single GPU via Megatron-style MLP + attention head splitting across 2× V100 SXM2 with NVLink.

A production-grade LLM inference server with a batching HTTP API, Redis request queue, Prometheus metrics, and Grafana dashboards — built on top of four progressively better GPU parallelism strategies, going from a 0.64x regression to a 1.05x speedup over single GPU.

**Model:** `mistralai/Mistral-7B-Instruct-v0.3` in fp16  
**Hardware:** 2× V100 SXM2 (32GB each, NVLink 154.7 GB/s) on Vast.ai  
**Reference:** Shoeybi et al., *Megatron-LM* (2019). [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)

---

## System Architecture

```mermaid
flowchart TD
    Client(["HTTP Clients"])
    Prometheus(["Prometheus"])
    Grafana(["Grafana"])

    subgraph Docker Compose
        API["FastAPI\nPOST /generate\nGET /health\nGET /metrics"]
        Redis[("Redis\nrequest queue")]
        Scheduler["Scheduler\n50ms batch window\nmax 8 requests"]

        subgraph Engine["Inference Engine"]
            GPU0["GPU 0\nweight shard"]
            GPU1["GPU 1\nweight shard"]
            GPU0 <-->|"NCCL all_reduce"| GPU1
        end
    end

    Client -->|HTTP| API
    API -->|enqueue + await Future| Redis
    Redis -->|dequeue batch| Scheduler
    Scheduler -->|engine.generate| Engine
    Engine -->|resolve Futures| Scheduler
    Prometheus -->|scrape /metrics| API
    Grafana -->|query| Prometheus
```

**Request lifecycle:**
1. Client POSTs to `/generate` — FastAPI enqueues it in Redis and returns a `Future`
2. Scheduler collects requests for 50ms (or until batch=8), then fires a batch
3. Engine tokenizes, runs a GPU forward pass across both GPUs in parallel, decodes
4. Futures resolve — all clients in the batch get their responses simultaneously
5. Prometheus scrapes `/metrics` every 15s — latency histograms, token throughput, queue depth, GPU memory

---

## Stack

| Layer | Technology | Detail |
|-------|-----------|--------|
| API | FastAPI | Async HTTP, Pydantic validation, lifespan model loading |
| Queue | Redis | Durable request queue, inspectable with `redis-cli` |
| Batching | asyncio + ThreadPoolExecutor | 50ms window, max batch=8, GIL-released GPU calls |
| Metrics | Prometheus + Grafana | p50/p99 latency, tok/s, queue depth, GPU memory |
| Inference | PyTorch fp16 | HuggingFace Transformers, custom tensor parallelism |
| GPU comm | NCCL / torchrun | Ring-allreduce, NVLink peer-to-peer |
| Deploy | Docker Compose | API + Redis + Prometheus + Grafana, one command |

---

## GPU Parallelism Results

Four implementations, each fixing one flaw in the previous:

| | Single GPU | col-parallel | Megatron `.to()` | NCCL MLP-only | **Full Megatron** |
|---|---|---|---|---|---|
| **req/s** | 1.10 | 0.70 | 0.81 | 1.04 | **1.15** |
| **p99 (ms)** | 926 | 1,530 | 1,312 | 1,020 | **922** |
| **vs single GPU** | 1.0x | 0.64x | 0.74x | 0.95x | **+1.05x** |

### The Progression

**Step 1 — Naive column-parallel: 0.64x**

Split every Linear weight in half, combine results with `.to()`. 450 cross-GPU transfers per forward pass, every one routed through the CPU driver.

```mermaid
flowchart LR
    x(["x (input)"])
    subgraph GPU0["GPU 0"]
        W0["W[:half]\npartial out_0"]
    end
    subgraph GPU1["GPU 1"]
        W1["W[half:]\npartial out_1"]
    end
    out(["output = cat(out_0, out_1)"])

    x --> W0
    x --> W1
    W0 -->|"out_0"| out
    W1 -->|"out_1.to(GPU0)\nCPU driver ~25µs"| out
```

2 transfers × 225 layers = **450 transfers/pass** — regression on every hardware config.

---

**Step 2 — Megatron alternating col/row: 0.74x**

Chain col-parallel → row-parallel so intermediate results feed directly into the next layer with zero gather. Only one all-reduce at the end of each MLP block.

```mermaid
flowchart LR
    x(["x"])
    subgraph GPU0["GPU 0"]
        G0["gate[:half]\nup[:half]"]
        H0["h_0 = act·up"]
        D0["down[:, :half]\npartial out_0"]
        G0 --> H0 --> D0
    end
    subgraph GPU1["GPU 1"]
        G1["gate[half:]\nup[half:]"]
        H1["h_1 = act·up"]
        D1["down[:, half:]\npartial out_1"]
        G1 --> H1 --> D1
    end
    out(["output = out_0 + out_1.to(GPU0)"])

    x --> G0
    x --> G1
    D0 --> out
    D1 -->|"1 transfer via CPU driver"| out
```

2 transfers per MLP block — **~224 transfers/pass**. Still regressing because `.to()` itself goes through the CPU driver regardless of transfer count.

---

**Step 3 — NCCL multi-process, MLP only: 0.95x**

One OS process per GPU. Replace `.to()` with `dist.all_reduce()` — NCCL ring-allreduce stays peer-to-peer in GPU SRAM, no CPU involved. MLP is now truly parallel. Attention is still replicated (both GPUs run identical computation).

```mermaid
flowchart LR
    x(["x (broadcast\nfrom rank 0)"])
    subgraph GPU0["Process 0 — cuda:0"]
        MLP0["MLP shard [:half]"]
        ATN0["Attention\n(full, replicated)"]
    end
    subgraph GPU1["Process 1 — cuda:1"]
        MLP1["MLP shard [half:]"]
        ATN1["Attention\n(full, replicated)"]
    end
    out(["output (rank 0)"])

    x --> MLP0
    x --> MLP1
    x --> ATN0
    x --> ATN1
    MLP0 -->|"dist.all_reduce()\nNVLink P2P ~10µs"| out
    MLP1 -->|"dist.all_reduce()"| out
    ATN0 --> out
```

Attention is ~35% of compute — leaving it replicated caps the speedup at 0.95x.

---

**Step 4 — Full Megatron, MLP + attention: 1.05x ✓**

Patch the attention module to split query heads (16/GPU) and KV heads (4/GPU). Mistral's GQA ratio (4:1) is preserved per rank, so attention math works identically. Both MLP and attention are now split — 2 GPUs finally beat 1.

```mermaid
flowchart LR
    x(["x (broadcast)"])
    subgraph GPU0["Process 0 — cuda:0"]
        A0["Attn\nq heads 0–15\nkv heads 0–3"]
        M0["MLP\nshard [:half]"]
        A0 --> M0
    end
    subgraph GPU1["Process 1 — cuda:1"]
        A1["Attn\nq heads 16–31\nkv heads 4–7"]
        M1["MLP\nshard [half:]"]
        A1 --> M1
    end
    out(["output (rank 0)"])

    x --> A0
    x --> A1
    A0 -->|"all_reduce o_proj"| out
    A1 -->|"all_reduce o_proj"| out
    M0 -->|"all_reduce down_proj"| out
    M1 -->|"all_reduce down_proj"| out
```

2 NCCL all_reduces per block × 32 blocks = 64 total — same count as step 3, but now attention compute is also halved.

---

## Concurrency Scaling

Tensor parallelism is a batching optimization — fixed communication overhead gets amortized as batch size grows.

| Concurrency | Single GPU req/s | Full Megatron req/s |
|------------|-----------------|---------------------|
| 1 | 0.21 | 0.15 |
| 2 | 0.27 | 0.28 |
| 4 | 0.58 | 0.56 |
| 8 | 1.17 | 1.11 |
| 16 | 2.13 | **2.21** |

At concurrency=1, Full Megatron is slower (NCCL setup overhead on a single request). By concurrency=2 they're even. At concurrency=16, Full Megatron pulls ahead. This is exactly why production systems use continuous batching — keeping the GPU saturated amortizes the communication cost.

---

## Running the Server

```bash
git clone https://github.com/sahilnale/distributed-llm-inference-server
cd distributed-llm-inference-server

# Spin up the full stack (API + Redis + Prometheus + Grafana)
cd single-process
HF_TOKEN=your_token docker compose up --build

# Send a request
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain attention in transformers", "max_tokens": 100}'

# View metrics
curl http://localhost:8000/metrics

# Grafana dashboard
open http://localhost:3000   # admin / admin
```

Switch to 2-GPU tensor parallel mode:
```bash
NUM_GPUS=2 docker compose up --build
```

## Running Benchmarks

```bash
export HF_TOKEN=your_token_here

# Single GPU baseline
python single-process/benchmarks/single_gpu.py

# Single-process tensor parallelism
python single-process/benchmarks/multi_gpu.py --mode column
python single-process/benchmarks/multi_gpu.py --mode megatron

# NCCL multi-process (MLP only)
torchrun --nproc_per_node=2 multi-process/benchmarks/benchmark.py

# Full Megatron (MLP + attention heads)
torchrun --nproc_per_node=2 full-megatron/benchmarks/benchmark.py
```

---

## Project Structure

```
distributed-llm-inference-server/
├── single-process/        Single Python process — full server stack + two parallelism modes
│   ├── src/
│   │   ├── server.py              FastAPI app, routes, lifespan model loading
│   │   ├── engine.py              Model loading, generate(), parallelism_mode param
│   │   ├── scheduler.py           Redis queue, 50ms batch window, asyncio Futures
│   │   ├── metrics.py             Prometheus counters, histograms, gauges
│   │   ├── parallel.py            Naive column-parallel (450 transfers/pass)
│   │   └── parallel_megatron.py   Megatron col/row (~224 transfers/pass)
│   ├── benchmarks/
│   ├── dashboard/grafana_config.json
│   ├── docker-compose.yml
│   ├── prometheus.yml
│   └── Dockerfile
├── multi-process/         One process per GPU — NCCL, MLP parallel only
│   ├── src/
│   │   ├── parallel_dist.py       MegatronMLPDist with dist.all_reduce()
│   │   └── engine.py              Rank-aware loading, input broadcast via NCCL
│   └── benchmarks/benchmark.py
└── full-megatron/         One process per GPU — NCCL, MLP + attention heads
    ├── src/
    │   ├── parallel_full.py       MegatronMLPFull + TensorParallelAttention
    │   └── engine.py
    └── benchmarks/benchmark.py
```

---

## Key Insights

- **`.to()` routes through the CPU driver regardless of hardware** — even on NVLink (154 GB/s), manual tensor copies go GPU → CPU → GPU. Switching to `dist.all_reduce()` (NCCL) was the single biggest win, jumping from 0.74x to 0.95x without changing the parallelism strategy at all.

- **Attention is ~35% of compute — replicating it caps speedup at 0.95x** — MLP-only tensor parallelism nearly closed the gap to single GPU, but both GPUs were still running identical attention computations. Splitting attention heads was the final step that crossed 1x.

- **Tensor parallelism is a batching optimization** — at concurrency=1, single GPU wins every time. The crossover is at concurrency=2. This is why production inference servers (vLLM, TGI) pair tensor parallelism with continuous batching — the communication overhead only makes sense when the GPU is saturated.

- **GQA head ratios must be preserved per rank** — Mistral uses Grouped Query Attention (32 query heads, 8 KV heads, 4:1 ratio). When splitting across 2 GPUs, each rank gets 16q + 4kv = still 4:1. If the ratio breaks, `repeat_kv` produces wrong-shaped tensors and the forward pass silently produces garbage.
