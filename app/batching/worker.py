"""
The dynamic batching worker - the centerpiece of this project.

Built from scratch with plain asyncio primitives (asyncio.Queue,
asyncio.Future, asyncio.wait_for) - no batching library, no framework
feature flag. This is exactly the design in docs/sequence_diagram.puml:

    Worker -> Queue : wait 10ms window, collect batch
    Worker -> Model : run inference (batched)
    Model --> Worker : batch output
    Worker -> Queue : fan-out results to each future

See docs/concepts/01_dynamic_batching.md for the full walkthrough of why
this exists and how the numbers behind it work.
"""

import asyncio
import time
from collections import defaultdict

import torch

from app import config
from app.batching.models import QueueItem
from app.inference.base import InferenceBackend


class BatchWorker:
    def __init__(self, request_queue: asyncio.Queue, backends: dict[str, InferenceBackend]):
        self.request_queue = request_queue
        self.backends = backends
        self._task: asyncio.Task | None = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run_forever())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_forever(self) -> None:
        while self._running:
            try:
                await self._collect_and_run_one_batch()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # A bug in one batch (a bad tensor shape, a backend crash)
                # must not kill the worker loop permanently - every request
                # after that would hang forever waiting on a future nothing
                # will ever resolve. Log it and keep serving.
                print(f"[BatchWorker] unexpected error, continuing: {exc}")

    async def _collect_and_run_one_batch(self) -> None:
        # Block here with no timeout: if there is nothing to do, there is no
        # reason to wake up and spin. This is what makes the worker
        # efficient when traffic is idle - it costs nothing while waiting.
        first_item = await self.request_queue.get()
        batch = [first_item]

        # The window starts counting from the moment the FIRST request in
        # this batch arrived, not from a fixed clock tick. This is why a
        # single request under low traffic still only waits ~10ms, not some
        # arbitrary alignment delay.
        window_deadline = time.monotonic() + config.BATCH_WINDOW_SECONDS

        while len(batch) < config.MAX_BATCH_SIZE:
            remaining = window_deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                # asyncio.wait_for races the queue.get() against the
                # remaining time left in the window. If another request
                # arrives in time, we grab it and loop again. If the
                # deadline passes first, wait_for raises TimeoutError and we
                # stop collecting.
                item = await asyncio.wait_for(self.request_queue.get(), timeout=remaining)
                batch.append(item)
            except asyncio.TimeoutError:
                break

        await self._run_batch(batch)

    async def _run_batch(self, batch: list[QueueItem]) -> None:
        # A single forward pass can only go through one model. If two
        # requests in this window asked for different backends, they can't
        # share a batch - group them and run one forward pass per backend
        # instead of one per request.
        groups: dict[str, list[QueueItem]] = defaultdict(list)
        for item in batch:
            groups[item.backend].append(item)

        for backend_name, items in groups.items():
            await self._run_group(backend_name, items)

    async def _run_group(self, backend_name: str, items: list[QueueItem]) -> None:
        batch_size = len(items)
        for item in items:
            item.batch_size = batch_size

        backend = self.backends[backend_name]
        input_tensor = torch.stack([item.image for item in items])

        try:
            # backend.predict() is a synchronous, CPU-heavy call - Phase 1's
            # benchmarks measured up to ~4 seconds for a batch of 16 on this
            # CPU. Calling it directly here would block the event loop for
            # that entire duration: no new HTTP connections accepted, no
            # other requests even reaching the queue. asyncio.to_thread runs
            # it on a worker thread instead, so the event loop stays free.
            # See docs/concepts/02_async_queue_processing.md for the full
            # explanation of why this matters.
            logits = await asyncio.to_thread(backend.predict, input_tensor)
        except Exception as exc:
            for item in items:
                if not item.future.done():
                    item.future.set_exception(exc)
            return

        for i, item in enumerate(items):
            if not item.future.done():
                item.future.set_result(logits[i])
