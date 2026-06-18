"""
FastAPI server — HTTP interface to the inference engine.

Endpoints:
  POST /generate  — submit a prompt, get a completion
  GET  /health    — liveness check
  GET  /metrics   — Prometheus metrics scrape endpoint

Everything heavy (model loading, batching) happens in engine.py and
scheduler.py. This file is intentionally thin — just HTTP plumbing.
"""

import logging
import os
import tempfile
import time

# Must set PROMETHEUS_MULTIPROC_DIR before importing prometheus_client.
# This tells the library where to write per-process metric files so they
# can be aggregated across uvicorn workers at scrape time.
_prom_dir = tempfile.mkdtemp(prefix="prom_")
os.environ.setdefault("PROMETHEUS_MULTIPROC_DIR", _prom_dir)

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

import metrics
from engine import InferenceEngine
from scheduler import RequestScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
# Pydantic models define the expected JSON shape.
# Field(default, ge=, le=) adds validation — FastAPI returns 422 automatically
# if a request violates these constraints. No manual validation needed.

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt text")
    max_tokens: int = Field(200, ge=1, le=2048, description="Maximum new tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    num_gpus: int
    gpu_memory: dict


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
# @asynccontextmanager lifespan replaces the old @app.on_event("startup").
# Code before `yield` runs at startup, code after runs at shutdown.
# The engine and scheduler are stored on app.state so route handlers can
# access them without globals.

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")

    model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
    num_gpus = int(os.environ.get("NUM_GPUS", 1))

    # Load model — this takes 30-60s on first run (downloads weights)
    engine = InferenceEngine(model_name=model_name, num_gpus=num_gpus)
    scheduler = RequestScheduler(engine=engine)
    await scheduler.start()

    app.state.engine = engine
    app.state.scheduler = scheduler

    logger.info("Server ready.")
    yield  # server runs here

    # Shutdown
    logger.info("Shutting down...")
    await scheduler.stop()


app = FastAPI(
    title="Distributed LLM Inference Server",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate a completion for the given prompt.

    The request is queued and batched with other concurrent requests before
    being sent to the GPU. Latency depends on queue depth and batch window.
    """
    start = time.perf_counter()

    try:
        text = await app.state.scheduler.submit(
            prompt=request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        metrics.record_request("success")
    except Exception as e:
        metrics.record_request("error")
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Rough token count — tokenizer not accessible here, word split is good enough
    # for the response metadata. Benchmarks use the engine's actual token counts.
    prompt_tokens = len(request.prompt.split())
    completion_tokens = len(text.split())

    metrics.record_latency(elapsed_ms / 1000)

    return GenerateResponse(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=round(elapsed_ms, 2),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Liveness check. Returns 200 if the server is up and the model is loaded.
    Kubernetes/Docker healthchecks hit this endpoint.
    """
    engine: InferenceEngine = app.state.engine
    return HealthResponse(
        status="ok",
        model=engine.model_name,
        num_gpus=engine.num_gpus,
        gpu_memory=engine.device_info,
    )


@app.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus scrape endpoint.

    Returns metrics in the Prometheus text exposition format.
    The Content-Type header tells Prometheus which parser to use.
    """
    registry = metrics.make_registry()
    data = generate_latest(registry)
    return PlainTextResponse(data, media_type=CONTENT_TYPE_LATEST)
