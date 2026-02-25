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
Tests for cache-related logging behavior.
"""
import logging
import pytest

from vgate.batcher import RequestBatcher
from vgate.cache import ResultCache


class MockBackend:
    """Mock inference backend for testing without GPU."""

    def __init__(self):
        self.call_count = 0

    def create_sampling_params(self, temperature, top_p, max_tokens):
        return {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}

    def generate(self, prompts, sampling_params):
        self.call_count += 1
        results = []
        for prompt in prompts:
            results.append({
                "text": f"Response to: {prompt[:30]}",
                "token_ids": list(range(10)),
                "num_tokens": 10,
                "metrics": {},
            })
        return results

    def shutdown(self):
        pass


class MockEngine:
    """Mock VGateEngine for testing."""

    def __init__(self):
        self.backend = MockBackend()


@pytest.mark.asyncio
async def test_cache_hit_emits_debug_log(caplog, monkeypatch):
    """Second identical request should emit 'Cache hit' debug log."""
    # Ensure propagation is enabled so caplog can capture records
    # (importing main.py sets propagate=False on the vgate root logger)
    logging.getLogger("vgate").propagate = True
    logging.getLogger("vgate.batcher").propagate = True

    batcher = RequestBatcher(
        engine=MockEngine(),
        max_batch_size=4,
        max_wait_time_ms=50.0,
    )

    expected_key = ResultCache.make_key(
        "Hello world",
        temperature=0.7,
        top_p=0.9,
        max_tokens=50,
    )[:8]

    await batcher.start()
    try:
        async def _fake_run_batch_inference(prompts, max_tokens, temperature=0.7, top_p=0.9):
            return [
                {"text": f"Response to: {p[:30]}", "ttft": 0.0, "tpot": 0.001, "total_tokens": 10}
                for p in prompts
            ]

        monkeypatch.setattr(batcher, "_run_batch_inference", _fake_run_batch_inference)

        with caplog.at_level(logging.DEBUG, logger="vgate.batcher"):
            await batcher.submit("Hello world", max_tokens=50, temperature=0.7, top_p=0.9)
            await batcher.submit("Hello world", max_tokens=50, temperature=0.7, top_p=0.9)
    finally:
        await batcher.stop()

    cache_hit_records = [r for r in caplog.records if r.name == "vgate.batcher" and r.message == "Cache hit"]

    assert cache_hit_records, "Expected at least one 'Cache hit' log record"
    assert cache_hit_records[-1].extra_data["cache_key"] == expected_key
