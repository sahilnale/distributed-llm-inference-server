# Distributed LLM Inference Server

Two implementations of 2-GPU tensor parallelism for LLM inference, built to understand why naive approaches regress and how production systems achieve real speedup.

Model: `mistralai/Mistral-7B-Instruct-v0.3` in fp16. Hardware: 2× V100 SXM2 (32GB, NVLink 154.7 GB/s).

## Implementations

| | [`src/`](src/) — Single Process | [`multi-process/`](multi-process/) — NCCL |
|---|---|---|
| **How** | One Python process, 2 GPUs | One process per GPU, `torchrun` |
| **All-reduce** | `.to()` + add (host-mediated DMA) | `dist.all_reduce()` (NCCL P2P) |
| **Parallelism** | Naive col-parallel OR Megatron col/row | Megatron col/row (MLP only) |
| **Launch** | `python benchmarks/multi_gpu.py` | `torchrun --nproc_per_node=2 ...` |
| **Speedup vs 1 GPU** | 0.64x–0.81x (regression) | TBD (expected >1x) |

**Key insight:** on NVLink hardware, the interconnect bandwidth (154 GB/s) isn't the bottleneck — *how you use it* is. `.to()` goes through the CPU driver and adds ~15–30μs per transfer regardless of hardware. NCCL stays peer-to-peer in GPU memory. Over 64 all-reduces per forward pass that difference compounds into measurable latency.

---

## Architecture

```mermaid
flowchart TD
    Client(["HTTP Clients\nPOST /generate\nGET /health\nGET /metrics"])
    Prometheus(["Prometheus\nscrapes /metrics"])
    Grafana(["Grafana\ndashboards"])

    subgraph Docker Compose
        API["FastAPI Server\nserver.py"]
        Metrics["/metrics endpoint"]
        Redis[("Redis\nqueue")]
        Scheduler["Scheduler\nbatch window 50ms / max 8"]

        subgraph InferenceEngine["Inference Engine — engine.py"]
            GPU0["GPU 0\n7 GB — W_left"]
            GPU1["GPU 1\n7 GB — W_right"]
            GPU0 -- "all-reduce\n(sum partial results)" --> GPU1
        end
    end

    Client -->|"POST /generate"| API
    API --> Metrics
    API --> Scheduler
    Scheduler -->|"enqueue"| Redis
    Redis -->|"dequeue batch"| Scheduler
    Scheduler -->|"engine.generate(batch)"| InferenceEngine
    Prometheus -->|"scrape"| Metrics
    Grafana -->|"query"| Prometheus
```

**Request flow:**
1. Client sends `POST /generate` with a prompt
2. FastAPI handler enqueues it in Redis and awaits a Future
3. Scheduler batch loop collects requests for 50ms (or until batch=8)
4. Engine runs one GPU forward pass on the whole batch
5. With 2 GPUs: weights are split across both GPUs (`parallelism_mode` controls strategy — column-parallel or Megatron col/row alternation)
6. Futures resolved, responses returned

---

## How Tensor Parallelism Works

Each transformer layer has 7 large weight matrices (Q, K, V, O projections + MLP gate/up/down). In single-GPU mode all 225 matrices sit on one card. In tensor parallel mode each matrix is split in half:

```
Single GPU                    Two GPUs (tensor parallel)
──────────                    ──────────────────────────
W (4096 × 4096)               GPU 0: W_0 (2048 × 4096)
all on cuda:0                 GPU 1: W_1 (2048 × 4096)

output = x @ W.T              out_0 = x @ W_0.T  (on GPU 0)
                              out_1 = x @ W_1.T  (on GPU 1)
                              output = cat([out_0, out_1])
```

Both GPUs load their half simultaneously — doubling effective memory bandwidth. This is why throughput improves: inference is memory-bandwidth-bound, not compute-bound.

---

## Benchmark Results

Hardware: **2× NVIDIA V100 SXM2 (32GB each, NVLink, 154.7 GB/s interconnect)** on Vast.ai

Model: `mistralai/Mistral-7B-Instruct-v0.3` in fp16 (13.5GB single GPU, 6.88GB + 6.62GB split across 2 GPUs)

### Experiment 1: Single GPU vs 2 GPU Throughput

| Metric | Single GPU (V100) | 2 GPU col-parallel (NVLink) | Speedup |
|--------|------------------|----------------------------|---------|
| Requests/sec | 1.1 | 0.7 | 0.64x |
| p99 latency (ms) | 926 | 1,530 | 0.60x |
| Tokens/sec | 149.5 | 95.2 | 0.64x |
| Scaling efficiency | — | — | 31.8% |

> **Finding:** Even on NVLink (154.7 GB/s), naive column-parallel is slower than single GPU. The 450 cross-GPU transfers per forward pass (2 per Linear × 225 layers) add ~4ms of synchronization overhead per batch that outweighs the memory bandwidth gain. This motivates the Megatron alternating col/row approach (~224 transfers) implemented in `parallel_megatron.py`.

### Experiment 2: Concurrency Scaling

| Concurrency | Single GPU req/s | 2 GPU req/s | Speedup |
|------------|-----------------|-------------|---------|
| 1 | 0.21 | 0.14 | 0.67x |
| 2 | 0.27 | 0.19 | 0.70x |
| 4 | 0.58 | 0.37 | 0.64x |
| 8 | 1.17 | 0.74 | 0.63x |
| 16 | 2.13 | 1.40 | 0.66x |

> **Finding:** Both configurations scale linearly with concurrency through batch=16 without saturating — the GPU still has headroom. Throughput grows ~10x from concurrency=1 to 16. The 2-GPU regression is consistent (~0.65x) across all batch sizes, confirming the bottleneck is per-forward-pass overhead, not queue or scheduling.

### Experiment 3: Batch Size Impact (Single GPU)

| Batch size | Req/s | Avg batch latency (ms) | Tokens/sec |
|-----------|-------|----------------------|-----------|
| 1 (no batching) | 0.18 | 5,470 | 36.4 |
| 2 | 0.32 | 6,324 | 63.4 |
| 4 | 0.61 | 6,600 | 121.2 |
| 8 | 1.17 | 6,847 | 233.6 |

> **Finding:** Going from batch=1 to batch=8 gives 6.4x throughput improvement with only 25% latency increase. Batching is almost free until the GPU saturates — the scheduler's 50ms batch window is well justified.

---

## Design Decisions

### Why tensor parallelism instead of pipeline parallelism?

Pipeline parallelism splits layers across GPUs (GPU 0 runs layers 1-16, GPU 1 runs layers 17-32). At low concurrency this creates idle time — GPU 1 waits for GPU 0 to finish before it can start. This "pipeline bubble" kills single-request latency.

Tensor parallelism splits each weight matrix across GPUs. Both GPUs work on every layer simultaneously. No idle time. The tradeoff is communication overhead — an all-reduce after every layer. Even on V100 NVLink (154.7 GB/s), naive column-parallel shows a regression (0.64x) because 450 transfers per forward pass is too many synchronization points regardless of bandwidth.

This is why production systems use Megatron-style alternating col/row parallelism — reducing MLP transfers from 6 to 2 per block brings total transfers from 450 to ~224. Implemented in `parallel_megatron.py`.

### Why Redis for the queue instead of an in-memory queue?

An in-memory asyncio queue would be simpler, but Redis gives us:
- **Durability** — requests survive server crashes
- **Observability** — you can inspect queue depth with `redis-cli llen inference:queue`
- **Scalability** — multiple server processes can share one queue

For a production inference server the Redis overhead is negligible compared to GPU time.

### Two tensor parallelism strategies — naive column-parallel vs Megatron

This project implements both approaches so the communication overhead is measurable, not just theoretical.

**Method 1: Naive column-parallel** (`src/parallel.py`)

Every Linear layer is treated the same: split output features across GPUs, gather results back.

```
For each Linear(in, out):
  GPU0: x → W0(out//2, in) → partial_0    (1 transfer in)
  GPU1: x → W1(out//2, in) → partial_1    (1 transfer in)
  output = cat([partial_0, partial_1])      (1 transfer out)
```

Cost: 2 transfers × 225 layers = **450 cross-GPU transfers per forward pass**.

**Method 2: Megatron-style alternating col/row** (`src/parallel_megatron.py`)

Reference: Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* (2019). [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)

The key insight from the paper: a column-parallel layer produces output **already split** across GPUs. A row-parallel layer expects input **already split** across GPUs. Back-to-back col→row layers need **zero communication** in the middle — only one all-reduce at the end of the pair.

Applied to the MLP block (gate/up/down projections):
```
gate_proj, up_proj: column-parallel  → output stays split across GPUs (NO gather)
down_proj:          row-parallel     → takes the split input, sums via all-reduce

GPU0: gate_0 = x @ G0.T  ||  GPU1: gate_1 = x @ G1.T   [no transfer between]
GPU0: up_0   = x @ U0.T  ||  GPU1: up_1   = x @ U1.T   [no transfer between]
GPU0: h_0 = act(gate_0)*up_0  ||  GPU1: h_1 = act(gate_1)*up_1  [no comm]
GPU0: out_0 = h_0 @ D0.T  ||  GPU1: out_1 = h_1 @ D1.T   [row parallel]
output = out_0 + out_1.to(GPU0)   [all-reduce — only 1 transfer out]
```

Cost per MLP block: **2 transfers** (copy x in, sum out) vs 6 in naive approach.

Total: ~64 MLP transfers + ~160 attention transfers = **~224 transfers per forward pass** — roughly 2× fewer syncs than naive.

The benchmarks quantify this gap. On NVLink (154.7 GB/s) the savings are real; on PCIe (12.3 GB/s) even the cheaper Megatron approach struggles because the interconnect itself is the bottleneck.

### Why fp16 instead of int8/int4 quantization?

fp16 halves memory vs fp32 with essentially no quality loss — it's a free lunch for inference. Quantization (int8/int4) goes further but introduces approximation error and requires calibration data. For a system focused on measuring parallelism, fp16 keeps the baseline clean.

---

## How to Reproduce on Vast.ai

### 1. Rent an instance

On [vast.ai](https://vast.ai), filter for:
- **GPU:** 2× A10G (or 2× RTX 3090 as a cheaper alternative)
- **Image:** `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`
- **Disk:** 50GB+ (model weights + Docker images)

### 2. SSH in and clone the repo

```bash
git clone <your-repo-url>
cd distributed-inference
```

### 3. Set your HuggingFace token

```bash
export HF_TOKEN=hf_your_token_here
```

You need to accept Llama's license at huggingface.co/meta-llama first.

### 4. Run benchmarks (no Docker needed)

```bash
pip install -r requirements.txt

# Single GPU baseline
python benchmarks/single_gpu.py

# Multi GPU with tensor parallelism
python benchmarks/multi_gpu.py

# Compare and print summary
python benchmarks/run_benchmarks.py --skip-single --skip-multi
```

Results are saved to `benchmarks/results/`.

### 5. Run the full server stack

```bash
# Spin up API + Redis + Prometheus + Grafana
docker compose up --build

# In another terminal, test it
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain attention in transformers", "max_tokens": 100}'

# Check metrics
curl http://localhost:8000/metrics

# Grafana dashboard
open http://localhost:3000   # admin / admin
```

### 6. Switch to 2-GPU mode

```bash
NUM_GPUS=2 docker compose up --build
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | required | HuggingFace access token |
| `MODEL_NAME` | `meta-llama/Llama-3.2-3B-Instruct` | Model to load |
| `NUM_GPUS` | `1` | Number of GPUs (1 or 2) |
| `BATCH_WINDOW_MS` | `50` | Max time to wait before firing a batch |
| `MAX_BATCH_SIZE` | `8` | Max requests per batch |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |

---

## Project Structure

```
distributed-inference/
├── src/
│   ├── server.py              FastAPI app, routes, lifespan
│   ├── engine.py              Model loading, generate() — parallelism_mode param
│   ├── parallel.py            Naive column-parallel (450 transfers/pass)
│   ├── parallel_megatron.py   Megatron-style col/row alternation (~224 transfers/pass)
│   ├── scheduler.py           Redis queue, batch window, Future resolution
│   └── metrics.py             Prometheus counters, histograms, gauges
├── benchmarks/
│   ├── single_gpu.py          Single GPU experiments
│   ├── multi_gpu.py           2-GPU experiments — runs both parallelism modes
│   ├── run_benchmarks.py      Runs all, computes speedup, prints summary
│   └── results/               JSON output files (one per mode)
├── dashboard/
│   └── grafana_config.json
├── Dockerfile
├── docker-compose.yml
├── prometheus.yml
└── requirements.txt
```
