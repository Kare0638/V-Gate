#!/usr/bin/env python3
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
Benchmark tool for comparing V-Gate inference backends.

Usage:
    python benchmarks/bench_compare.py --backends vllm sglang
    python benchmarks/bench_compare.py --backends vllm --rounds 5 --max-tokens 256
    VGATE_DRY_RUN=true python benchmarks/bench_compare.py --backends vllm sglang
"""

import argparse
import json
import statistics
import sys
import time

from vgate.config import ModelConfig, get_config
from vgate.engine import VGateEngine


DEFAULT_PROMPTS = [
    "Explain the concept of machine learning in one paragraph.",
    "Write a Python function that computes the Fibonacci sequence.",
    "What are the benefits of using a load balancer?",
]


def run_benchmark(engine_type, model_id, prompts, max_tokens, warmup, rounds):
    """Run benchmark for a single backend, return aggregated stats."""
    model_config = ModelConfig(engine_type=engine_type, model_id=model_id)
    engine = VGateEngine(model_config=model_config)

    # Warmup
    print(f"  Warming up ({warmup} rounds)...")
    for _ in range(warmup):
        for prompt in prompts:
            engine.chat_completions(prompt, max_tokens=max_tokens)

    # Test rounds
    latencies = []
    ttfts = []
    tpots = []
    token_counts = []

    print(f"  Running {rounds} test rounds...")
    for r in range(rounds):
        round_start = time.perf_counter()
        round_tokens = 0
        for prompt in prompts:
            result = engine.chat_completions(prompt, max_tokens=max_tokens)
            ttfts.append(result["ttft"])
            tpots.append(result["tpot"])
            round_tokens += result["total_tokens"]
        round_latency = time.perf_counter() - round_start
        latencies.append(round_latency)
        token_counts.append(round_tokens)

    # Shutdown backend
    engine.backend.shutdown()

    total_tokens = sum(token_counts)
    total_time = sum(latencies)

    def percentile(data, pct):
        s = sorted(data)
        idx = int(len(s) * pct / 100)
        idx = min(idx, len(s) - 1)
        return s[idx]

    stats = {
        "engine_type": engine_type,
        "rounds": rounds,
        "prompts_per_round": len(prompts),
        "latency": {
            "mean_s": round(statistics.mean(latencies), 4),
            "p50_s": round(statistics.median(latencies), 4),
            "p95_s": round(percentile(latencies, 95), 4),
        },
        "ttft": {
            "mean_s": round(statistics.mean(ttfts), 4) if ttfts else 0,
            "p50_s": round(statistics.median(ttfts), 4) if ttfts else 0,
            "p95_s": round(percentile(ttfts, 95), 4) if ttfts else 0,
        },
        "tpot": {
            "mean_s": round(statistics.mean(tpots), 4) if tpots else 0,
            "p50_s": round(statistics.median(tpots), 4) if tpots else 0,
            "p95_s": round(percentile(tpots, 95), 4) if tpots else 0,
        },
        "throughput": {
            "total_tokens": total_tokens,
            "tokens_per_second": round(total_tokens / total_time, 2) if total_time > 0 else 0,
        },
    }
    return stats


def print_table(all_stats):
    """Print a comparison table to the terminal."""
    header = f"{'Metric':<30}"
    for s in all_stats:
        header += f"  {s['engine_type']:>12}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    rows = [
        ("Rounds", lambda s: str(s["rounds"])),
        ("Prompts/round", lambda s: str(s["prompts_per_round"])),
        ("Latency mean (s)", lambda s: f"{s['latency']['mean_s']:.4f}"),
        ("Latency p50 (s)", lambda s: f"{s['latency']['p50_s']:.4f}"),
        ("Latency p95 (s)", lambda s: f"{s['latency']['p95_s']:.4f}"),
        ("TTFT mean (s)", lambda s: f"{s['ttft']['mean_s']:.4f}"),
        ("TTFT p50 (s)", lambda s: f"{s['ttft']['p50_s']:.4f}"),
        ("TTFT p95 (s)", lambda s: f"{s['ttft']['p95_s']:.4f}"),
        ("TPOT mean (s)", lambda s: f"{s['tpot']['mean_s']:.4f}"),
        ("TPOT p50 (s)", lambda s: f"{s['tpot']['p50_s']:.4f}"),
        ("TPOT p95 (s)", lambda s: f"{s['tpot']['p95_s']:.4f}"),
        ("Total tokens", lambda s: str(s["throughput"]["total_tokens"])),
        ("Tokens/sec", lambda s: f"{s['throughput']['tokens_per_second']:.2f}"),
    ]

    for label, fn in rows:
        row = f"{label:<30}"
        for s in all_stats:
            row += f"  {fn(s):>12}"
        print(row)

    print("=" * len(header) + "\n")


def main():
    parser = argparse.ArgumentParser(description="V-Gate backend benchmark tool")
    parser.add_argument(
        "--backends", nargs="+", default=["vllm"],
        choices=["vllm", "sglang"],
        help="Backend(s) to benchmark (default: vllm)",
    )
    parser.add_argument("--model", default=None, help="Model ID (uses config default)")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens per generation")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup rounds")
    parser.add_argument("--rounds", type=int, default=3, help="Test rounds")
    parser.add_argument("--output", choices=["table", "json"], default="table", help="Output format")
    args = parser.parse_args()

    config = get_config()
    model_id = args.model or config.model.model_id

    all_stats = []
    for backend_name in args.backends:
        print(f"\n--- Benchmarking: {backend_name} ---")
        stats = run_benchmark(
            engine_type=backend_name,
            model_id=model_id,
            prompts=DEFAULT_PROMPTS,
            max_tokens=args.max_tokens,
            warmup=args.warmup,
            rounds=args.rounds,
        )
        all_stats.append(stats)

    if args.output == "json":
        print(json.dumps(all_stats, indent=2))
    else:
        print_table(all_stats)


if __name__ == "__main__":
    main()
