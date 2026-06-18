# Distributed LLM Inference Server

A distributed inference system that splits a Llama model across 2 GPUs using tensor parallelism, exposes a batching HTTP API, and benchmarks single vs multi-GPU performance.

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
5. With 2 GPUs: each Linear layer is split across both GPUs, partial results summed via all-reduce
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

Two hardware configurations tested on Vast.ai:
- **2× NVIDIA A10 (22GB each, PCIe 4.0, 12.3 GB/s interconnect)** — shows communication bottleneck
- **2× NVIDIA V100 (16GB each, NVLink, 154.7 GB/s interconnect)** — shows speedup with fast interconnect *(pending)*

Model: `mistralai/Mistral-7B-Instruct-v0.3` in fp16 (13.5GB single GPU, 6.88GB + 6.62GB split across 2 GPUs)

### Experiment 1: Single GPU vs 2 GPU Throughput

| Metric | Single GPU (A10) | 2 GPU (A10, PCIe) | Speedup |
|--------|-----------------|-------------------|---------|
| Requests/sec | 0.95 | 0.70 | 0.74x |
| Avg latency (ms) | 1,011 | 1,372 | 0.74x |
| p99 latency (ms) | 1,080 | 1,541 | — |
| Tokens/sec | 130.9 | 95.4 | 0.73x |
| Scaling efficiency | — | — | 36.8% |

> **Finding:** 2-GPU with PCIe interconnect (12.3 GB/s) is slower than single GPU. The all-reduce after every Linear layer costs more time than the bandwidth gain from splitting weights. NVLink (154.7 GB/s) is required for positive speedup — results pending on V100 NVLink instance.

### Experiment 2: Concurrency Scaling

| Concurrency | Single GPU req/s | 2 GPU req/s (PCIe) | Single GPU tok/s | 2 GPU tok/s |
|------------|-----------------|-------------------|-----------------|------------|
| 1 | 0.19 | 0.14 | 37.9 | 28.3 |
| 2 | 0.27 | 0.19 | 53.7 | 37.7 |
| 4 | 0.52 | 0.37 | 104.3 | 74.7 |
| 8 | 0.99 | 0.73 | 197.6 | 146.7 |
| 16 | 1.78 | 1.34 | 356.9 | 268.1 |

> **Finding:** Both configurations scale linearly with concurrency up to batch=16 without saturating — the GPU still has headroom. Throughput increases 9x (single) and 9.6x (multi) from concurrency=1 to 16, while latency only increases ~70%. The GPU is underutilized at low concurrency regardless of configuration.

### Experiment 3: Batch Size Impact (Single GPU)

| Batch size | Req/s | Avg batch latency (ms) | Tokens/sec |
|-----------|-------|----------------------|-----------|
| 1 (no batching) | 0.15 | 6,826 | 29.3 |
| 2 | 0.27 | 7,389 | 54.1 |
| 4 | 0.52 | 7,678 | 104.2 |
| 8 | 0.99 | 8,098 | 197.6 |

> **Finding:** Going from batch=1 to batch=8 gives 6.7x throughput improvement (29.3 → 197.6 tok/s) with only 19% latency increase (6.8s → 8.1s). Batching is almost free until the GPU saturates — the scheduler's 50ms batch window is well justified.

---

## Design Decisions

### Why tensor parallelism instead of pipeline parallelism?

Pipeline parallelism splits layers across GPUs (GPU 0 runs layers 1-16, GPU 1 runs layers 17-32). At low concurrency this creates idle time — GPU 1 waits for GPU 0 to finish before it can start. This "pipeline bubble" kills single-request latency.

Tensor parallelism splits each weight matrix across GPUs. Both GPUs work on every layer simultaneously. No idle time. The tradeoff is communication overhead — an all-reduce after every layer. On NVLink (154.7 GB/s) this is negligible. On PCIe (12.3 GB/s) it costs more than the bandwidth gain — as our A10 benchmark shows (0.73x speedup, i.e. a regression).

This is why production tensor parallelism deployments exclusively use NVLink-connected hardware (A100 SXM, H100 SXM). The V100 NVLink benchmark will confirm the positive speedup on fast interconnect.

### Why Redis for the queue instead of an in-memory queue?

An in-memory asyncio queue would be simpler, but Redis gives us:
- **Durability** — requests survive server crashes
- **Observability** — you can inspect queue depth with `redis-cli llen inference:queue`
- **Scalability** — multiple server processes can share one queue

For a production inference server the Redis overhead is negligible compared to GPU time.

### Why column parallelism vs Megatron-style alternating col/row?

We use column parallelism for every Linear layer (split output features, concatenate results). Megatron-LM alternates column and row parallel layers so the output of one feeds directly into the next without a gather step — halving inter-GPU communication.

We chose column-only because it's easier to reason about and implement correctly. The benchmark will show ~80-90% scaling efficiency rather than ~95%. That gap is worth explaining: it quantifies the communication overhead and demonstrates why production systems invest in the more complex alternating approach.

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
│   ├── server.py       FastAPI app, routes, lifespan
│   ├── engine.py       Model loading, generate() method
│   ├── parallel.py     Tensor parallelism — splits Linear layers across GPUs
│   ├── scheduler.py    Redis queue, batch window, Future resolution
│   └── metrics.py      Prometheus counters, histograms, gauges
├── benchmarks/
│   ├── single_gpu.py   Single GPU experiments
│   ├── multi_gpu.py    2-GPU experiments (reuses single_gpu functions)
│   ├── run_benchmarks.py  Runs both, computes speedup, prints summary
│   └── results/        JSON output files
├── dashboard/
│   └── grafana_config.json
├── Dockerfile
├── docker-compose.yml
├── prometheus.yml
└── requirements.txt
```
