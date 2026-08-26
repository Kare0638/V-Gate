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
The request deadline.

RequestBatcher.submit() has accepted a timeout since it was written, and
nothing ever passed one -- the parameter had tests, the production path did
not use it, and a request could wait for an admission permit indefinitely. The
tests here are about that wiring and about which HTTP status it produces.
"""

import asyncio

import pytest

from vgate.batcher import RequestBatcher
from vgate.config import ReliabilityConfig


class SlowBackend:
    """A backend whose every call takes longer than any deadline under test."""

    supports_concurrent_calls = False  # forces admission to one at a time

    def __init__(self, delay=5.0):
        self.delay = delay
        self.calls = 0

    def create_sampling_params(self, temperature, top_p, max_tokens):
        return {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}

    def generate(self, prompts, sampling_params):
        self.calls += 1
        import time
        time.sleep(self.delay)
        return [{"text": "slow", "token_ids": [1], "num_tokens": 1, "metrics": {}}
                for _ in prompts]


class SlowEngine:
    def __init__(self, delay=5.0):
        self.backend = SlowBackend(delay)

    def chat_completions_batch(self, prompts, max_tokens=256, temperature=0.7, top_p=0.9):
        return self.backend.generate(prompts, None)


@pytest.fixture
async def batcher():
    b = RequestBatcher(engine=SlowEngine())
    await b.start()
    yield b
    b._running = False


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

def test_default_matches_the_worker_timeout():
    """
    Not an arbitrary number. The gateway's deadline lines up with the timeout
    it applies to a worker call, so the two do not disagree about how long a
    request is allowed to take.
    """
    from vgate.config import WorkerConfig
    assert ReliabilityConfig().request_timeout_seconds == WorkerConfig().timeout_seconds


def test_zero_disables_the_deadline():
    """`or None` in the request path turns 0 into "no timeout"."""
    assert (ReliabilityConfig(request_timeout_seconds=0).request_timeout_seconds or None) is None


def test_a_negative_deadline_is_rejected():
    with pytest.raises(ValueError, match="must be >= 0"):
        ReliabilityConfig(request_timeout_seconds=-1)


# --------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_deadline_covers_the_wait_for_a_permit():
    """
    The point of the change. A request that never reaches the backend -- because
    something else holds the only admission permit -- must still time out.
    Bounding the inference alone would leave the queue unbounded in time, which
    is exactly the state this fixes.
    """
    b = RequestBatcher(engine=SlowEngine(delay=3.0))
    await b.start()
    try:
        # Occupies the single permit for the whole test.
        blocker = asyncio.create_task(b.submit("blocking prompt"))
        await asyncio.sleep(0.2)

        started = asyncio.get_running_loop().time()
        with pytest.raises(asyncio.TimeoutError):
            await b.submit("queued behind the blocker", timeout=0.3)
        waited = asyncio.get_running_loop().time() - started

        assert waited < 1.5, f"waited {waited:.1f}s; the deadline did not cover the queue"
        blocker.cancel()
        await asyncio.gather(blocker, return_exceptions=True)
    finally:
        b._running = False


@pytest.mark.asyncio
async def test_no_deadline_means_the_request_waits():
    """The disabled case still behaves as it did, rather than silently capping."""
    b = RequestBatcher(engine=SlowEngine(delay=0.4))
    await b.start()
    try:
        result = await b.submit("unbounded", timeout=None)
        assert result["text"] == "slow"
    finally:
        b._running = False


@pytest.mark.asyncio
async def test_a_timed_out_request_does_not_cancel_work_others_need():
    """
    Abandonment policy, restated because the deadline is what now triggers it:
    a caller giving up must not cancel an inference another caller is still
    waiting on, and must not discard a result already being paid for.
    """
    b = RequestBatcher(engine=SlowEngine(delay=0.6))
    await b.start()
    try:
        patient = asyncio.create_task(b.submit("shared prompt"))
        await asyncio.sleep(0.1)

        with pytest.raises(asyncio.TimeoutError):
            await b.submit("shared prompt", timeout=0.2)

        assert (await patient)["text"] == "slow", "the impatient caller killed shared work"
    finally:
        b._running = False


# --------------------------------------------------------------------------
# HTTP surface
# --------------------------------------------------------------------------

def test_exceeding_the_deadline_returns_504_not_503():
    """
    The status code is the part clients act on, and 503 and 504 mean different
    things. 503 with Retry-After says capacity is unavailable -- back off and
    come back. 504 says this particular request was accepted and the work
    behind it did not finish in time; nothing here knows that retrying sooner
    would help, so no Retry-After is offered.

    Sharing one code for both would leave a client unable to tell "the pool is
    down" from "my request was slow".
    """
    import main
    from fastapi.testclient import TestClient

    class TimingOutBatcher:
        async def submit(self, *args, **kwargs):
            raise asyncio.TimeoutError()

        async def start(self):
            pass

        async def stop(self):
            pass

    original = main.batcher
    main.batcher = TimingOutBatcher()
    try:
        with TestClient(main.app) as client:
            main.batcher = TimingOutBatcher()  # lifespan replaces it on startup
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "will not finish"}],
                    "max_tokens": 8,
                },
            )
    finally:
        main.batcher = original

    assert response.status_code == 504, response.text
    assert "Retry-After" not in response.headers, (
        "504 must not advertise a retry delay; nothing here knows one"
    )


def test_the_request_path_passes_the_configured_deadline():
    """
    The wiring itself, which is all this change actually is.

    Written after a negative test exposed the gap: removing `timeout=timeout`
    from the request path broke nothing, because the batcher tests supply their
    own deadline and the status-code test uses a stub that raises regardless.
    Both passed against a gateway that had gone back to waiting forever.
    """
    import main
    from fastapi.testclient import TestClient

    seen = {}

    class RecordingBatcher:
        async def submit(self, *args, **kwargs):
            seen["timeout"] = kwargs.get("timeout", "<not passed>")
            return {"text": "ok", "total_tokens": 1, "prompt_tokens": 0}

        async def start(self):
            pass

        async def stop(self):
            pass

    original = main.batcher
    try:
        with TestClient(main.app) as client:
            main.batcher = RecordingBatcher()
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 8,
                },
            )
    finally:
        main.batcher = original

    assert seen["timeout"] == main.config.reliability.request_timeout_seconds, (
        f"request path supplied {seen['timeout']!r}; a deadline that never "
        "reaches submit() leaves the queue unbounded in time"
    )
