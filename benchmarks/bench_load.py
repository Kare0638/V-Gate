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
Concurrent load benchmark against a *running* V-Gate server.

Unlike bench_compare.py (which drives the engine in-process), this tool
exercises the full HTTP path: security middleware, batcher, cache, and
backend. It measures client-observed latency percentiles and pulls
batch/cache/queue counters from /stats to show what the server did with
the load.

Usage:
    # Start a server first, e.g.:
    #   VGATE_DRY_RUN=true python main.py
    python benchmarks/bench_load.py --concurrency 8 --requests 80
    python benchmarks/bench_load.py --prompt-file prompts.txt --output json
    python benchmarks/bench_load.py --stream --concurrency 8 --requests 40
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

DEFAULT_PROMPTS = [
    "Explain the concept of machine learning in one paragraph.",
    "Write a Python function that computes the Fibonacci sequence.",
    "What are the benefits of using a load balancer?",
    "Summarize the CAP theorem in two sentences.",
]


def _percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = min(int(len(s) * pct / 100), len(s) - 1)
    return s[idx]


async def _get_json(session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
    async with session.get(url) as resp:
        return await resp.json()


async def _get_text(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(url) as resp:
        return await resp.text()


def _parse_counter_value(metrics_text: str, metric_name: str) -> float:
    """Extract an unlabeled Prometheus counter's value from /metrics text
    exposition output, e.g. a "vgate_stream_tokens_total 150.0" line."""
    prefix = metric_name + " "
    for line in metrics_text.splitlines():
        if line.startswith(prefix):
            return float(line.split()[1])
    return 0.0


async def _send_one(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int,
    headers: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    payload = {
        "model": "vgate-bench",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    start = time.perf_counter()
    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            body = await resp.json()
            ok = resp.status == 200 and "choices" in body
    except Exception:
        ok = False
        body = {}
    latency = time.perf_counter() - start
    tokens = body.get("usage", {}).get("completion_tokens", 0) if ok else 0
    return {"latency_s": latency, "ok": ok, "tokens": tokens, "ttft_s": None}


async def _send_one_stream(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int,
    headers: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    """Like _send_one, but consumes an SSE stream instead of a single JSON
    body. Returns `content_chunks` — the number of content-delta *events*,
    not a token count. V-Gate's SSE chunks don't carry a per-request final
    token count (no `stream_options: {include_usage: true}` support yet),
    and a single delta can carry more than one real token, so this number
    isn't comparable across backends/scheduling and must never be reported
    as tokens/sec — see run_load_test, which instead diffs the server's own
    vgate_stream_tokens_total counter for that.
    """
    payload = {
        "model": "vgate-bench",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    start = time.perf_counter()
    ttft: Optional[float] = None
    chunk_count = 0
    ok = False
    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                return {"latency_s": time.perf_counter() - start, "ok": False, "content_chunks": 0, "ttft_s": None}
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    ok = True
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if "error" in data:
                    ok = False
                    break
                delta = data.get("choices", [{}])[0].get("delta", {})
                if delta.get("content"):
                    chunk_count += 1
                    if ttft is None:
                        ttft = time.perf_counter() - start
    except Exception:
        ok = False
    latency = time.perf_counter() - start
    return {"latency_s": latency, "ok": ok, "content_chunks": chunk_count, "ttft_s": ttft}


async def run_load_test(
    base_url: str,
    concurrency: int,
    total_requests: int,
    prompts: List[str],
    max_tokens: int = 64,
    api_key: Optional[str] = None,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    Fire `total_requests` chat completion requests at `concurrency` concurrent
    workers against a running server, then diff /stats before and after to
    attribute batching/cache behavior to this run.

    With `stream=True`, each request is sent with `"stream": true` and
    consumed as SSE instead of a single JSON response, and the result
    includes client-observed TTFT percentiles. Streaming bypasses
    RequestBatcher (see main.py's _stream_chat_completion), so the
    `batching`/`cache` sections reflect only what non-streaming traffic did
    during this run — for an all-streaming run they'll show no activity at
    all, which is expected, not a bug. Token counts for streaming come from
    diffing the server's own vgate_stream_tokens_total counter (/metrics)
    rather than counting SSE content-delta events client-side, since a
    single delta can carry more than one real token.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    chat_url = f"{base_url}/v1/chat/completions"
    send_one = _send_one_stream if stream else _send_one

    async with aiohttp.ClientSession() as session:
        stats_before = await _get_json(session, f"{base_url}/stats")
        stream_tokens_before = (
            _parse_counter_value(await _get_text(session, f"{base_url}/metrics"), "vgate_stream_tokens_total")
            if stream else 0.0
        )

        sem = asyncio.Semaphore(concurrency)

        async def _bounded(i: int) -> Dict[str, Any]:
            async with sem:
                return await send_one(
                    session, chat_url, prompts[i % len(prompts)], max_tokens, headers
                )

        wall_start = time.perf_counter()
        results = await asyncio.gather(*[_bounded(i) for i in range(total_requests)])
        wall_time = time.perf_counter() - wall_start

        stats_after = await _get_json(session, f"{base_url}/stats")
        stream_tokens_after = (
            _parse_counter_value(await _get_text(session, f"{base_url}/metrics"), "vgate_stream_tokens_total")
            if stream else 0.0
        )

    latencies = [r["latency_s"] for r in results if r["ok"]]
    failures = sum(1 for r in results if not r["ok"])
    if stream:
        total_tokens = int(round(stream_tokens_after - stream_tokens_before))
        total_content_chunks = sum(r["content_chunks"] for r in results)
    else:
        total_tokens = sum(r["tokens"] for r in results)
        total_content_chunks = None

    before_b, after_b = stats_before["batcher"], stats_after["batcher"]
    before_c, after_c = stats_before["cache"], stats_after["cache"]

    new_requests = after_b["total_requests"] - before_b["total_requests"]
    new_batches = after_b["total_batches"] - before_b["total_batches"]
    new_dedup = after_b["total_deduplicated"] - before_b["total_deduplicated"]
    new_hits = after_c["hits"] - before_c["hits"]
    new_misses = after_c["misses"] - before_c["misses"]

    latency_stats = {
        "mean_s": round(statistics.mean(latencies), 4) if latencies else 0,
        "p50_s": round(_percentile(latencies, 50), 4),
        "p95_s": round(_percentile(latencies, 95), 4),
        "p99_s": round(_percentile(latencies, 99), 4),
        "max_s": round(max(latencies), 4) if latencies else 0,
    }
    if stream:
        ttfts = [r["ttft_s"] for r in results if r["ok"] and r.get("ttft_s") is not None]
        latency_stats["ttft_mean_s"] = round(statistics.mean(ttfts), 4) if ttfts else 0
        latency_stats["ttft_p50_s"] = round(_percentile(ttfts, 50), 4)
        latency_stats["ttft_p95_s"] = round(_percentile(ttfts, 95), 4)

    return {
        "config": {
            "concurrency": concurrency,
            "total_requests": total_requests,
            "unique_prompts": len(prompts),
            "max_tokens": max_tokens,
            "stream": stream,
        },
        "wall_time_s": round(wall_time, 4),
        "failures": failures,
        "latency": latency_stats,
        "throughput": {
            "total_tokens": total_tokens,
            "tokens_per_second": round(total_tokens / wall_time, 2) if wall_time > 0 else 0,
            "requests_per_second": round(total_requests / wall_time, 2) if wall_time > 0 else 0,
            **({"content_chunks": total_content_chunks} if stream else {}),
        },
        "batching": {
            "requests": new_requests,
            "batches": new_batches,
            "average_batch_size": round(new_requests / new_batches, 2) if new_batches > 0 else 0,
            "deduplicated": new_dedup,
            "avg_queue_time_s": after_b.get("avg_queue_time_s", 0),
            "avg_ttft_s": after_b.get("avg_ttft_s", 0),
            "avg_tpot_s": after_b.get("avg_tpot_s", 0),
        },
        "cache": {
            "hits": new_hits,
            "misses": new_misses,
            "hit_rate": (
                round(new_hits / (new_hits + new_misses), 4)
                if (new_hits + new_misses) > 0 else 0
            ),
        },
    }


def load_prompts(prompt_file: Optional[str]) -> List[str]:
    if not prompt_file:
        return DEFAULT_PROMPTS
    lines = [
        line.strip()
        for line in Path(prompt_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return lines or DEFAULT_PROMPTS


def format_markdown(result: Dict[str, Any], title: str = "V-Gate Load Benchmark") -> str:
    c, lat, thr = result["config"], result["latency"], result["throughput"]
    b, ch = result["batching"], result["cache"]
    stream_mode = c.get("stream", False)
    lines = [
        f"## {title}",
        "",
        f"- Concurrency: {c['concurrency']}",
        f"- Total requests: {c['total_requests']}",
        f"- Unique prompts: {c['unique_prompts']}",
        f"- Max tokens: {c['max_tokens']}",
        f"- Mode: {'streaming (SSE)' if stream_mode else 'non-streaming'}",
        f"- Failures: {result['failures']}",
        f"- Wall time: {result['wall_time_s']}s",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Latency mean (s) | {lat['mean_s']} |",
        f"| Latency p50 (s) | {lat['p50_s']} |",
        f"| Latency p95 (s) | {lat['p95_s']} |",
        f"| Latency p99 (s) | {lat['p99_s']} |",
        f"| Latency max (s) | {lat['max_s']} |",
    ]
    if stream_mode:
        lines += [
            f"| Client-observed TTFT mean (s) | {lat.get('ttft_mean_s', 0)} |",
            f"| Client-observed TTFT p50 (s) | {lat.get('ttft_p50_s', 0)} |",
            f"| Client-observed TTFT p95 (s) | {lat.get('ttft_p95_s', 0)} |",
        ]
    lines += [
        f"| Tokens/sec{' (server-reported, via vgate_stream_tokens_total)' if stream_mode else ''} | {thr['tokens_per_second']} |",
    ]
    if stream_mode and "content_chunks" in thr:
        lines.append(
            f"| SSE content-delta events (diagnostic only — not a token count) | {thr['content_chunks']} |"
        )
    lines += [
        f"| Requests/sec | {thr['requests_per_second']} |",
        f"| Avg batch size | {b['average_batch_size']} |",
        f"| Batches formed | {b['batches']} |",
        f"| Deduplicated requests | {b['deduplicated']} |",
        f"| Avg queue time (s) | {b['avg_queue_time_s']} |",
        f"| Avg TTFT (s, batcher-reported, non-streaming only) | {b['avg_ttft_s']} |",
        f"| Avg TPOT (s, batcher-reported, non-streaming only) | {b['avg_tpot_s']} |",
        f"| Cache hit rate (this run) | {ch['hit_rate']} |",
        f"| Cache hits / misses (this run) | {ch['hits']} / {ch['misses']} |",
        "",
    ]
    return "\n".join(lines)


async def _main_async(args: argparse.Namespace) -> Dict[str, Any]:
    prompts = load_prompts(args.prompt_file)
    return await run_load_test(
        base_url=args.url,
        concurrency=args.concurrency,
        total_requests=args.requests,
        prompts=prompts,
        max_tokens=args.max_tokens,
        api_key=args.api_key,
        stream=args.stream,
    )


def main():
    parser = argparse.ArgumentParser(
        description="V-Gate concurrent load benchmark (hits a running server)"
    )
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of a running V-Gate server")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent in-flight requests")
    parser.add_argument("--requests", type=int, default=80, help="Total number of requests to send")
    parser.add_argument("--prompt-file", default=None, help="Path to a text file with one prompt per line")
    parser.add_argument("--max-tokens", type=int, default=64, help="max_tokens per request")
    parser.add_argument("--api-key", default=None, help="Bearer API key, if server security is enabled")
    parser.add_argument(
        "--stream", action="store_true",
        help=(
            "Send stream=true and consume SSE. Reports client-observed TTFT "
            "percentiles; bypasses RequestBatcher (see ROADMAP.md Phase 2), so "
            "batching/cache stats reflect only concurrent non-streaming traffic."
        ),
    )
    parser.add_argument("--output", choices=["json", "markdown"], default="markdown")
    args = parser.parse_args()

    result = asyncio.run(_main_async(args))

    if args.output == "json":
        print(json.dumps(result, indent=2))
    else:
        print(format_markdown(result, title=f"Load Benchmark: {args.url}"))


if __name__ == "__main__":
    main()
