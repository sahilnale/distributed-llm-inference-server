"""
Request scheduler — batches incoming requests before sending to the engine.

Why batching matters:
  One GPU forward pass with batch=8 costs roughly the same as batch=1.
  The scheduler collects requests for BATCH_WINDOW_MS milliseconds (or until
  MAX_BATCH_SIZE is reached), then fires them all at the engine together.
  This trades a small amount of latency for a large throughput gain.

How it works:
  1. HTTP handler calls scheduler.submit(prompt, ...) and awaits the result.
  2. submit() pushes the request into a Redis queue and stores an asyncio.Future
     in a local dict keyed by request ID.
  3. A background loop (_batch_loop) runs forever:
       - Drains up to MAX_BATCH_SIZE items from Redis
       - Waits up to BATCH_WINDOW_MS for more to arrive
       - Calls engine.generate() on the batch
       - Resolves each Future with its corresponding output
  4. submit()'s await unblocks and returns the result to the HTTP handler.

asyncio.Future vs threading:
  Everything here is async (single-threaded event loop). The engine.generate()
  call is CPU/GPU-bound and would block the event loop, so we run it in a
  ThreadPoolExecutor — that lets the event loop keep accepting requests while
  the GPU is crunching.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import redis.asyncio as aioredis

import metrics

logger = logging.getLogger(__name__)

BATCH_WINDOW_MS = int(os.environ.get("BATCH_WINDOW_MS", 50))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", 8))
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
QUEUE_KEY = "inference:queue"


class RequestScheduler:
    def __init__(self, engine):
        self.engine = engine

        # Maps request_id → asyncio.Future
        # When the batch loop resolves a request, it sets the Future's result,
        # which unblocks the corresponding submit() call.
        self._pending: dict[str, asyncio.Future] = {}

        # ThreadPoolExecutor for running engine.generate() without blocking
        # the event loop. One thread is enough — the GPU is the bottleneck,
        # not the thread count.
        self._executor = ThreadPoolExecutor(max_workers=1)

        self._redis: aioredis.Redis | None = None
        self._loop_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Connect to Redis and launch the background batch loop."""
        self._redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
        await self._redis.ping()
        logger.info(f"Scheduler connected to Redis at {REDIS_URL}")

        # asyncio.create_task() schedules _batch_loop() to run concurrently
        # on the same event loop. It runs alongside request handlers — not
        # in a separate thread.
        self._loop_task = asyncio.create_task(self._batch_loop())
        logger.info(f"Batch loop started (window={BATCH_WINDOW_MS}ms, max_batch={MAX_BATCH_SIZE})")

    async def stop(self):
        """Graceful shutdown — cancel the loop and close Redis."""
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        if self._redis:
            await self._redis.aclose()
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Public interface (called by server.py)
    # ------------------------------------------------------------------

    async def submit(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """
        Enqueue a request and wait for the result.

        Returns the generated text string. Raises on engine error.

        The caller (HTTP handler) awaits this coroutine. It suspends here
        until _batch_loop() resolves the Future for this request_id.
        """
        request_id = str(uuid.uuid4())

        # Create a Future on the current event loop. We'll resolve it later
        # from _batch_loop when the engine returns results.
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[request_id] = future

        # Serialize the request and push to Redis list (RPUSH = push to right/tail)
        payload = json.dumps({
            "request_id": request_id,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        })
        await self._redis.rpush(QUEUE_KEY, payload)

        # Update queue depth metric
        depth = await self._redis.llen(QUEUE_KEY)
        metrics.set_queue_depth(depth)

        logger.debug(f"Enqueued request {request_id[:8]}")

        # Suspend here until batch loop calls future.set_result()
        return await future

    # ------------------------------------------------------------------
    # Background batch loop
    # ------------------------------------------------------------------

    async def _batch_loop(self):
        """
        Runs forever. Each iteration:
          1. Block-wait for at least one item in the queue (BLPOP)
          2. Collect more items up to MAX_BATCH_SIZE within BATCH_WINDOW_MS
          3. Run inference
          4. Resolve futures
        """
        logger.info("Batch loop running")

        while True:
            try:
                batch = await self._collect_batch()
                if not batch:
                    continue

                metrics.record_batch(len(batch))
                metrics.update_gpu_memory()

                await self._run_batch(batch)

            except asyncio.CancelledError:
                raise  # propagate so stop() can clean up
            except Exception:
                logger.exception("Error in batch loop — continuing")

    async def _collect_batch(self) -> list[dict]:
        """
        Wait for the first request, then greedily collect more.

        BLPOP blocks until an item appears (timeout=0 = wait forever).
        After the first item arrives, we have BATCH_WINDOW_MS to grab more
        before we fire the batch regardless.

        This is the core of the batching strategy:
          - Never fire an empty batch (BLPOP ensures at least one item)
          - Never wait more than BATCH_WINDOW_MS (latency cap)
          - Never exceed MAX_BATCH_SIZE (VRAM cap)
        """
        # BLPOP returns (key, value) or None on timeout
        result = await self._redis.blpop(QUEUE_KEY, timeout=1)
        if result is None:
            return []

        _, raw = result
        batch = [json.loads(raw)]

        # Collect more within the window
        deadline = time.monotonic() + (BATCH_WINDOW_MS / 1000)

        while len(batch) < MAX_BATCH_SIZE:
            remaining_ms = (deadline - time.monotonic()) * 1000
            if remaining_ms <= 0:
                break

            # Non-blocking pop (LPOP) — grab next item if available
            raw = await self._redis.lpop(QUEUE_KEY)
            if raw is None:
                # Nothing yet — yield control briefly and check again
                await asyncio.sleep(remaining_ms / 1000 / 2)
                continue

            batch.append(json.loads(raw))

        metrics.set_queue_depth(await self._redis.llen(QUEUE_KEY))
        logger.debug(f"Collected batch of {len(batch)}")
        return batch

    async def _run_batch(self, batch: list[dict]) -> None:
        """
        Run engine.generate() for the batch and resolve each Future.

        engine.generate() is synchronous (blocks until GPU is done).
        We run it in the ThreadPoolExecutor so the event loop stays
        responsive to new incoming requests during inference.
        """
        prompts = [item["prompt"] for item in batch]
        max_new_tokens = batch[0]["max_new_tokens"]   # use first request's params
        temperature = batch[0]["temperature"]

        start = time.perf_counter()

        loop = asyncio.get_event_loop()
        try:
            # run_in_executor(executor, fn, *args) runs fn(*args) in a thread
            # and returns an awaitable. The event loop is free while the GPU runs.
            outputs: list[str] = await loop.run_in_executor(
                self._executor,
                self.engine.generate,
                prompts,
                max_new_tokens,
                temperature,
            )
        except Exception as e:
            # Resolve all futures with the exception so HTTP handlers return 500
            for item in batch:
                fut = self._pending.pop(item["request_id"], None)
                if fut and not fut.done():
                    fut.set_exception(e)
            return

        elapsed = time.perf_counter() - start
        total_tokens = sum(len(o.split()) for o in outputs) * max_new_tokens
        tps = total_tokens / elapsed if elapsed > 0 else 0

        metrics.record_latency(elapsed)
        metrics.record_tokens_per_second(tps)

        # Resolve each future with its corresponding output
        for item, output in zip(batch, outputs):
            request_id = item["request_id"]
            fut = self._pending.pop(request_id, None)
            if fut and not fut.done():
                fut.set_result(output)

        logger.debug(f"Batch done: {len(batch)} requests in {elapsed:.2f}s ({tps:.0f} tok/s)")
