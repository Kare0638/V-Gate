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
Unit tests for benchmarks/bench_load.py's pure aggregation/formatting logic.

The concurrent HTTP load path (run_load_test) needs a real running server
(aiohttp requires a real socket, unlike httpx's ASGI transport) and is
exercised end-to-end by benchmarks/run_report.py; it is not re-tested here
to keep this suite fast and GPU-free.
"""

from benchmarks.bench_load import _percentile, format_markdown, load_prompts, DEFAULT_PROMPTS


class TestPercentile:
    def test_empty(self):
        assert _percentile([], 50) == 0.0

    def test_single_value(self):
        assert _percentile([1.5], 99) == 1.5

    def test_p50_matches_median_for_odd_length(self):
        assert _percentile([1.0, 2.0, 3.0], 50) == 2.0

    def test_p99_clamps_to_max(self):
        data = [float(i) for i in range(10)]
        assert _percentile(data, 99) == max(data)


class TestLoadPrompts:
    def test_no_file_returns_defaults(self):
        assert load_prompts(None) == DEFAULT_PROMPTS

    def test_reads_nonempty_lines(self, tmp_path):
        p = tmp_path / "prompts.txt"
        p.write_text("first prompt\n\n  second prompt  \n", encoding="utf-8")
        assert load_prompts(str(p)) == ["first prompt", "second prompt"]

    def test_blank_file_falls_back_to_defaults(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("\n\n  \n", encoding="utf-8")
        assert load_prompts(str(p)) == DEFAULT_PROMPTS


class TestFormatMarkdown:
    def _sample_result(self):
        return {
            "config": {"concurrency": 4, "total_requests": 20, "unique_prompts": 3, "max_tokens": 64},
            "wall_time_s": 1.234,
            "failures": 0,
            "latency": {"mean_s": 0.1, "p50_s": 0.09, "p95_s": 0.2, "p99_s": 0.25, "max_s": 0.3},
            "throughput": {"total_tokens": 100, "tokens_per_second": 81.0, "requests_per_second": 16.0},
            "batching": {
                "requests": 20, "batches": 4, "average_batch_size": 5.0, "deduplicated": 2,
                "avg_queue_time_s": 0.01, "avg_ttft_s": 0.0, "avg_tpot_s": 0.005,
            },
            "cache": {"hits": 5, "misses": 15, "hit_rate": 0.25},
        }

    def test_contains_title_and_key_metrics(self):
        md = format_markdown(self._sample_result(), title="My Scenario")
        assert "## My Scenario" in md
        assert "0.1" in md  # latency mean
        assert "81.0" in md  # tokens/sec
        assert "0.25" in md  # cache hit rate

    def test_no_failures_data_missing_keyerror(self):
        # Guard against KeyError regressions when a caller passes a
        # freshly-built result dict from run_load_test.
        result = self._sample_result()
        md = format_markdown(result)
        assert isinstance(md, str) and len(md) > 0
