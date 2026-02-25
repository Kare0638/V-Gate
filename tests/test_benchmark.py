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
Tests for benchmark tools.
"""

import os
from unittest.mock import patch

import pytest

from vgate.config import reset_config


@pytest.fixture(autouse=True)
def _reset():
    reset_config()
    yield
    reset_config()


class TestBenchmarkCLI:
    """Test benchmark CLI aggregation logic in DRY_RUN mode."""

    def test_run_benchmark_dry_run(self):
        from benchmarks.bench_compare import run_benchmark

        stats = run_benchmark(
            engine_type="vllm",
            model_id="test-model",
            prompts=["Hello"],
            max_tokens=32,
            warmup=0,
            rounds=2,
        )

        assert stats["engine_type"] == "vllm"
        assert stats["rounds"] == 2
        assert stats["prompts_per_round"] == 1
        assert "latency" in stats
        assert "ttft" in stats
        assert "tpot" in stats
        assert "throughput" in stats
        assert stats["throughput"]["total_tokens"] > 0
        assert stats["latency"]["mean_s"] >= 0

    def test_run_benchmark_sglang_dry_run(self):
        from benchmarks.bench_compare import run_benchmark

        stats = run_benchmark(
            engine_type="sglang",
            model_id="test-model",
            prompts=["Test prompt"],
            max_tokens=64,
            warmup=1,
            rounds=2,
        )

        assert stats["engine_type"] == "sglang"
        assert stats["throughput"]["total_tokens"] > 0

    def test_print_table(self, capsys):
        from benchmarks.bench_compare import print_table

        stats = [
            {
                "engine_type": "vllm",
                "rounds": 2,
                "prompts_per_round": 1,
                "latency": {"mean_s": 0.1, "p50_s": 0.1, "p95_s": 0.12},
                "ttft": {"mean_s": 0.01, "p50_s": 0.01, "p95_s": 0.02},
                "tpot": {"mean_s": 0.005, "p50_s": 0.005, "p95_s": 0.006},
                "throughput": {"total_tokens": 100, "tokens_per_second": 1000.0},
            }
        ]
        print_table(stats)
        output = capsys.readouterr().out
        assert "vllm" in output
        assert "Latency mean" in output


class TestBenchmarkEndpoint:
    """Test /v1/benchmark API endpoint."""

    @pytest.mark.asyncio
    async def test_benchmark_endpoint_default(self):
        from httpx import ASGITransport, AsyncClient
        from main import app, lifespan

        async with lifespan(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/v1/benchmark", json={})

            assert response.status_code == 200
            data = response.json()
            assert "engine_type" in data
            assert "latency" in data
            assert "throughput" in data
            assert data["rounds"] == 3  # default

    @pytest.mark.asyncio
    async def test_benchmark_endpoint_custom(self):
        from httpx import ASGITransport, AsyncClient
        from main import app, lifespan

        async with lifespan(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/v1/benchmark", json={
                    "prompts": ["Hello"],
                    "max_tokens": 32,
                    "rounds": 1,
                })

            assert response.status_code == 200
            data = response.json()
            assert data["rounds"] == 1
            assert data["prompts_per_round"] == 1
