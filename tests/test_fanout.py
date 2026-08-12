# Copyright 2025 the V-Gate authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for the dedup / admission / fan-out behavior of RequestBatcher.

The property that matters for routing is that each request reaches the
backend as its own call. The property that matters for cost is that
identical concurrent requests reach it only once between them.
"""

import asyncio
import time

import pytest

from vgate.batcher import RequestBatcher


class _RecordingBackend:
    """Records every generate() call so tests can assert on dispatch shape."""

    supports_concurrent_calls = True

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.calls = []

    def create_sampling_params(self, temperature, top_p, max_tokens):
        return {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}

    def generate(self, prompts, sampling_params):
        self.calls.append((list(prompts), dict(sampling_params)))
        if self.delay:
            time.sleep(self.delay)
        return [
            {"text": f"out:{p}", "token_ids": list(range(3)), "num_tokens": 3, "metrics": {}}
            for p in prompts
        ]

    def shutdown(self):
        pass


class _Engine:
    def __init__(self, backend):
        self.backend = backend


async def _batcher(backend, **kwargs) -> RequestBatcher:
    b = RequestBatcher(engine=_Engine(backend), **kwargs)
    await b.start()
    return b


# --------------------------------------------------------------------------
# Fan-out
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_each_request_is_its_own_backend_call():
    """
    The routing-granularity property: distinct requests must not be merged
    into one backend call, or a router could only place them as a group.
    """
    backend = _RecordingBackend()
    b = await _batcher(backend, max_batch_size=8)
    try:
        await asyncio.gather(*(b.submit(f"p{i}", max_tokens=4) for i in range(5)))
    finally:
        await b.stop()

    assert len(backend.calls) == 5, "expected one backend call per request"
    for prompts, _ in backend.calls:
        assert len(prompts) == 1, f"a call carried {len(prompts)} prompts"


@pytest.mark.asyncio
async def test_differing_sampling_params_stay_separate():
    """Different params were the reason batches had to be split before; with
    fan-out each request carries its own params by construction."""
    backend = _RecordingBackend()
    b = await _batcher(backend, max_batch_size=8)
    try:
        await asyncio.gather(
            b.submit("same", max_tokens=4, temperature=0.1),
            b.submit("same", max_tokens=4, temperature=0.9),
        )
    finally:
        await b.stop()

    assert len(backend.calls) == 2
    temps = sorted(params["temperature"] for _, params in backend.calls)
    assert temps == [0.1, 0.9], "params must not be shared across requests"


# --------------------------------------------------------------------------
# In-flight deduplication
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_identical_concurrent_requests_share_one_inference():
    backend = _RecordingBackend(delay=0.05)
    b = await _batcher(backend, max_batch_size=8)
    try:
        results = await asyncio.gather(
            *(b.submit("duplicate", max_tokens=4) for _ in range(5))
        )
    finally:
        await b.stop()

    assert len(backend.calls) == 1, "5 identical requests should cost one inference"
    assert all(r["text"] == "out:duplicate" for r in results)
    assert b.get_metrics()["total_deduplicated"] == 4


@pytest.mark.asyncio
async def test_dedup_is_not_limited_to_a_time_window():
    """
    Coalescing follows the in-flight request, not a fixed window.

    Under the old model two identical requests only merged if they landed in
    the same batching window. Here the second one merges as long as the first
    is still running, however late it arrives.
    """
    backend = _RecordingBackend(delay=0.2)
    b = await _batcher(backend, max_batch_size=8)
    try:
        first = asyncio.create_task(b.submit("slow", max_tokens=4))
        # Far longer than any old batching window would have been.
        await asyncio.sleep(0.1)
        second = asyncio.create_task(b.submit("slow", max_tokens=4))
        await asyncio.gather(first, second)
    finally:
        await b.stop()

    assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_sequential_identical_requests_hit_cache_not_dedup():
    """Once the first completes, the second is a cache hit, not a coalesce."""
    backend = _RecordingBackend()
    b = await _batcher(backend, max_batch_size=8)
    try:
        await b.submit("once", max_tokens=4)
        await b.submit("once", max_tokens=4)
    finally:
        await b.stop()

    assert len(backend.calls) == 1
    stats = b.get_metrics()
    assert stats["total_deduplicated"] == 0
    assert stats["cache"]["hits"] >= 1


@pytest.mark.asyncio
async def test_failed_inference_propagates_to_all_waiters():
    """A shared failure must reach every coalesced caller, not just the leader."""

    class _FailingBackend(_RecordingBackend):
        def generate(self, prompts, sampling_params):
            time.sleep(0.03)
            raise RuntimeError("backend exploded")

    backend = _FailingBackend()
    b = await _batcher(backend, max_batch_size=8)
    try:
        results = await asyncio.gather(
            *(b.submit("doomed", max_tokens=4) for _ in range(3)),
            return_exceptions=True,
        )
    finally:
        await b.stop()

    assert len(results) == 3
    assert all(isinstance(r, RuntimeError) for r in results), results


@pytest.mark.asyncio
async def test_inflight_map_is_empty_after_completion():
    backend = _RecordingBackend()
    b = await _batcher(backend, max_batch_size=4)
    try:
        await asyncio.gather(*(b.submit(f"x{i}", max_tokens=4) for i in range(4)))
        assert b._inflight == {}
    finally:
        await b.stop()


@pytest.mark.asyncio
async def test_failed_inference_is_not_cached():
    """A failure must not poison the cache for later retries."""

    state = {"fail": True}

    class _FlakyBackend(_RecordingBackend):
        def generate(self, prompts, sampling_params):
            self.calls.append((list(prompts), dict(sampling_params)))
            if state["fail"]:
                raise RuntimeError("transient")
            return [
                {"text": f"out:{p}", "token_ids": [1], "num_tokens": 1, "metrics": {}}
                for p in prompts
            ]

    backend = _FlakyBackend()
    b = await _batcher(backend, max_batch_size=4)
    try:
        with pytest.raises(RuntimeError):
            await b.submit("retryable", max_tokens=4)

        state["fail"] = False
        result = await b.submit("retryable", max_tokens=4)
        assert result["text"] == "out:retryable"
    finally:
        await b.stop()

    assert len(backend.calls) == 2, "the retry must actually reach the backend"


# --------------------------------------------------------------------------
# Admission
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_deduplicated_waiters_do_not_consume_admission_permits():
    """
    Followers wait on someone else's inference, so they must not hold a
    permit -- otherwise N duplicates of one prompt would exhaust admission
    while only one inference is actually running.
    """
    backend = _RecordingBackend(delay=0.1)
    b = await _batcher(backend, max_batch_size=2)
    try:
        dupes = [asyncio.create_task(b.submit("shared", max_tokens=4)) for _ in range(5)]
        await asyncio.sleep(0.03)

        # One permit is held by the leader; a different prompt must still be
        # admitted rather than queueing behind the four followers.
        other = await asyncio.wait_for(b.submit("different", max_tokens=4), timeout=1.0)
        assert other["text"] == "out:different"

        await asyncio.gather(*dupes)
    finally:
        await b.stop()

    assert len(backend.calls) == 2
