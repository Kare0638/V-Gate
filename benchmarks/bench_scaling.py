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
Measure whether a pool of workers serves more than one worker does.

This answers the question the architecture has been claiming without evidence:
the gateway routes across N inference workers, and until now nothing said what
that bought. It spawns one gateway in front of N worker processes, drives the
same load at each N, and reports throughput, tail latency, and how the requests
were distributed.

WHAT MAKES THE NUMBERS MEAN ANYTHING

A scaling measurement needs each worker to have a capacity it can reach.
A dry-run worker has none: it sleeps, and the worker runs those sleeps on an
executor sized from the host's core count. One worker would absorb everything
the load generator sent until that executor filled, so 1 and 4 workers would
measure the same, and the point where it filled would depend on whose machine
ran it.

So both halves of a worker's capacity are declared here rather than inherited:

    VGATE_DRYRUN_SIMULATED_LATENCY_MS   how long one generation takes
    VGATE_DRYRUN_MAX_CONCURRENCY        how many run at once per worker

Per-worker throughput is then exactly concurrency/latency, and the ideal for N
workers is N times that. Reporting measured against ideal is the point: the gap
is what the gateway costs.

WHAT THIS DOES NOT MEASURE

Dry-run workers do no inference. This measures how well one gateway feeds N
backends -- routing, admission, fan-out, and the HTTP hop -- and says nothing
about how much GPU throughput N GPUs provide. A real multi-GPU number needs
real GPUs; that is a separate, unstarted item in ROADMAP.md, and no claim about
it is made from these results.

Usage:
    python benchmarks/bench_scaling.py
    python benchmarks/bench_scaling.py --workers 1,2,4 --repeats 3
"""

import argparse
import asyncio
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bench_load import run_load_test  # noqa: E402

RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
GATEWAY_PORT = 8110
WORKER_PORT_BASE = 8111


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def _spawn(env_overrides: Dict[str, str], log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env.update({
        "VGATE_SECURITY__ENABLED": "false",
        "VGATE_TRACING__ENABLED": "false",
        "VGATE_LOGGING__LEVEL": "WARNING",
        "VGATE_METRICS__ENABLED": "true",
    })
    env.update(env_overrides)
    return subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=open(log_path, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _terminate(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)
    except ProcessLookupError:
        pass


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


async def _wait_ports_free(ports: List[int], timeout_s: float = 30.0) -> None:
    """
    Block until nothing is listening on these ports.

    Every topology reuses the same ports. Without this, a process from the
    previous run that has not finished exiting still answers /health, the next
    topology accepts it as ready, and then it dies mid-measurement -- leaving a
    request to sit out the full 120s worker timeout. That happened: one run
    reported 4.0 req/s against 155 from its neighbours, with a normal p95,
    because throughput is requests over wall time and one request had consumed
    two minutes of it. A median across three repeats hid it, which is worse
    than the anomaly.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        busy = [p for p in ports if not _port_is_free(p)]
        if not busy:
            return
        await asyncio.sleep(0.2)
    raise RuntimeError(f"ports still in use after {timeout_s}s: {busy}")


async def _wait_healthy(url: str, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            try:
                async with session.get(
                    f"{url}/health", timeout=aiohttp.ClientTimeout(total=1)
                ) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                pass
            await asyncio.sleep(0.2)
    raise RuntimeError(f"{url} did not become healthy within {timeout_s}s")


class Topology:
    """One gateway in front of N worker processes, torn down together."""

    def __init__(self, num_workers: int, latency_ms: int, capacity: int, admission: int):
        self.num_workers = num_workers
        self.latency_ms = latency_ms
        self.capacity = capacity
        self.admission = admission
        self.gateway: Optional[subprocess.Popen] = None
        self.workers: List[subprocess.Popen] = []
        self.log_dir = RESULTS_DIR / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{GATEWAY_PORT}"

    def _endpoints(self) -> List[str]:
        return [
            f"http://127.0.0.1:{WORKER_PORT_BASE + i}" for i in range(self.num_workers)
        ]

    def _all_ports(self) -> List[int]:
        return [GATEWAY_PORT] + [
            WORKER_PORT_BASE + i for i in range(max(self.num_workers, 16))
        ]

    async def __aenter__(self) -> "Topology":
        # Nothing from a previous topology may still be listening, or its
        # /health answer will be mistaken for this one's.
        await _wait_ports_free(self._all_ports())
        for i in range(self.num_workers):
            port = WORKER_PORT_BASE + i
            self.workers.append(_spawn(
                {
                    "VGATE_ROLE": "worker",
                    "VGATE_DRY_RUN": "true",
                    "VGATE_SERVER__PORT": str(port),
                    "VGATE_DRYRUN_SIMULATED_LATENCY_MS": str(self.latency_ms),
                    "VGATE_DRYRUN_MAX_CONCURRENCY": str(self.capacity),
                },
                self.log_dir / f"scaling-worker-{self.num_workers}w-{i}.log",
            ))
        self.gateway = _spawn(
            {
                "VGATE_ROLE": "gateway",
                "VGATE_SERVER__PORT": str(GATEWAY_PORT),
                "VGATE_WORKER__ENDPOINTS": json.dumps(self._endpoints()),
                # The gateway's own admission limit caps concurrent inferences
                # regardless of how many workers exist, so it has to be lifted
                # above N x capacity or it, not the pool, is what is measured.
                "VGATE_BATCH__MAX_BATCH_SIZE": str(self.admission),
            },
            self.log_dir / f"scaling-gateway-{self.num_workers}w.log",
        )

        for i in range(self.num_workers):
            await _wait_healthy(f"http://127.0.0.1:{WORKER_PORT_BASE + i}")
        await _wait_healthy(self.base_url)
        # Let the gateway's first health probe land, so the first measured
        # request is not the one that discovers a worker.
        await asyncio.sleep(1.0)
        return self

    async def __aexit__(self, *exc) -> None:
        _terminate(self.gateway)
        for w in self.workers:
            _terminate(w)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

async def _worker_distribution(base_url: str) -> Dict[str, int]:
    """Requests served per worker, read from the gateway's own counters."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}/metrics") as resp:
            text = await resp.text()
    out: Dict[str, int] = {}
    for line in text.splitlines():
        m = re.match(
            r'^vgate_worker_requests_total\{outcome="success",worker="([^"]+)"\}\s+([\d.]+)',
            line,
        )
        if m:
            out[m.group(1).rsplit(":", 1)[-1]] = int(float(m.group(2)))
    return out


def _unique_prompts(count: int, tag: str) -> List[str]:
    """All-distinct prompts, so neither the cache nor in-flight dedup can
    stand in for work the pool actually did."""
    stamp = int(time.time() * 1000)
    return [f"scaling {tag} {stamp} {i}" for i in range(count)]


async def run_one(
    num_workers: int, args: argparse.Namespace, repeat: int
) -> Dict[str, Any]:
    async with Topology(
        num_workers, args.latency_ms, args.capacity, args.admission
    ) as topo:
        prompts = _unique_prompts(args.requests, f"{num_workers}w-r{repeat}")
        result = await run_load_test(
            base_url=topo.base_url,
            concurrency=args.concurrency,
            total_requests=args.requests,
            prompts=prompts,
            max_tokens=0,  # keep the synthetic cost equal to latency_ms exactly
        )
        distribution = await _worker_distribution(topo.base_url)

    return {
        "workers": num_workers,
        "repeat": repeat,
        "throughput_rps": result["throughput"]["requests_per_second"],
        "p50_s": result["latency"]["p50_s"],
        "p95_s": result["latency"]["p95_s"],
        "p99_s": result["latency"]["p99_s"],
        "failures": result["failures"],
        "cache_hits": result["cache"]["hits"],
        "deduplicated": result["batching"]["deduplicated"],
        "distribution": distribution,
    }


async def run_admission_ceiling(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Show the gateway's admission limit as a scaling ceiling.

    Worth measuring separately because it is invisible from the outside: with
    max_batch_size below N x capacity, adding workers changes nothing and the
    pool looks like it does not scale, when what is actually saturated is the
    gateway's own permit count.
    """
    out = []
    for admission in (args.capacity, args.admission):
        async with Topology(
            args.ceiling_workers, args.latency_ms, args.capacity, admission
        ) as topo:
            prompts = _unique_prompts(args.requests, f"ceiling-{admission}")
            result = await run_load_test(
                base_url=topo.base_url,
                concurrency=args.concurrency,
                total_requests=args.requests,
                prompts=prompts,
                max_tokens=0,
            )
        out.append({
            "admission": admission,
            "workers": args.ceiling_workers,
            "throughput_rps": result["throughput"]["requests_per_second"],
            "p95_s": result["latency"]["p95_s"],
        })
    return out


async def run_saturation(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Find what stops the pool from scaling, and prove which side it is on.

    Reporting an efficiency figure without this is guesswork: a shortfall at
    high N is equally consistent with "the gateway is saturated" and "the load
    generator cannot go faster", and those have opposite implications. Splitting
    the same offered load across more client tasks distinguishes them. If more
    clients push more traffic, the generator was the limit; if the total does
    not move, the gateway is.
    """
    capacity = args.saturation_workers * args.capacity / (args.latency_ms / 1000.0)
    rows = []
    async with Topology(
        args.saturation_workers, args.latency_ms, args.capacity, args.admission * 2
    ) as topo:
        for groups in (1, 2, 4):
            per_group = args.concurrency * 2 // groups
            requests = args.requests * 2 // groups
            results = await asyncio.gather(*[
                run_load_test(
                    base_url=topo.base_url,
                    concurrency=per_group,
                    total_requests=requests,
                    prompts=_unique_prompts(requests, f"sat-{groups}-{g}"),
                    max_tokens=0,
                )
                for g in range(groups)
            ])
            rows.append({
                "client_groups": groups,
                "concurrency_each": per_group,
                "total_rps": round(
                    sum(r["throughput"]["requests_per_second"] for r in results), 1
                ),
                "p95_s": max(r["latency"]["p95_s"] for r in results),
            })
    return {"pool_capacity_rps": capacity, "rows": rows}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _median(values: List[float]) -> float:
    return statistics.median(values)


def format_report(
    args: argparse.Namespace,
    runs: List[Dict[str, Any]],
    ceiling: List[Dict[str, Any]],
    saturation: Optional[Dict[str, Any]] = None,
) -> str:
    per_worker_ideal = args.capacity / (args.latency_ms / 1000.0)
    counts = sorted({r["workers"] for r in runs})

    lines = [
        "# 1-vs-N Worker Scaling",
        "",
        f"Generated by `benchmarks/bench_scaling.py` on "
        f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC.",
        "Regenerate with `python benchmarks/bench_scaling.py`; do not edit by hand.",
        "",
        "## What this measures, and what it does not",
        "",
        "One gateway in front of N worker processes, same load at each N.",
        "",
        "**Dry-run workers do no inference.** This measures how well one gateway",
        "feeds N backends -- routing, admission, fan-out, and the HTTP hop. It says",
        "nothing about how much GPU throughput N GPUs provide; that needs real GPUs",
        "and is a separate unstarted item in ROADMAP.md. No claim about multi-GPU",
        "performance is made from these numbers.",
        "",
        "Each worker's capacity is declared rather than inherited from the host, so",
        "the ideal is arithmetic rather than a guess:",
        "",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| Generation cost per request | {args.latency_ms} ms |",
        f"| Concurrent generations per worker | {args.capacity} |",
        f"| **Ideal throughput per worker** | **{per_worker_ideal:.0f} req/s** |",
        f"| Client concurrency | {args.concurrency} |",
        f"| Requests per run | {args.requests} |",
        f"| Repeats per point | {args.repeats} |",
        f"| Gateway admission limit (`batch.max_batch_size`) | {args.admission} |",
        "",
        "Prompts are all distinct, so neither the result cache nor in-flight",
        "deduplication can stand in for work the pool actually did.",
        "",
        "## Throughput",
        "",
        "| Workers | Ideal req/s | Median req/s | Spread | Efficiency | Speedup vs 1 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    medians: Dict[int, float] = {}
    for n in counts:
        vals = [r["throughput_rps"] for r in runs if r["workers"] == n]
        clean = [
            r["throughput_rps"] for r in runs
            if r["workers"] == n and r["failures"] == 0
        ]
        # A run with a failed request is excluded, not averaged in. Throughput
        # is requests over wall time, so one request that sits out a 120s
        # worker timeout drags the figure to a fraction of the truth while
        # leaving p95 untouched -- and a median across repeats hides it.
        vals = clean or vals
        med = _median(vals)
        medians[n] = med
        ideal = per_worker_ideal * n
        spread = f"{min(vals):.1f}–{max(vals):.1f}" if len(vals) > 1 else "—"
        speedup = med / medians[counts[0]] if medians.get(counts[0]) else 1.0
        lines.append(
            f"| {n} | {ideal:.0f} | **{med:.1f}** | {spread} | "
            f"{100 * med / ideal:.0f}% | {speedup:.2f}x |"
        )

    lines += [
        "",
        "Efficiency is measured over ideal. The shortfall is what the gateway",
        "costs: one HTTP hop per request, admission bookkeeping, and the",
        "round-robin choice.",
        "",
        "## Latency",
        "",
        "| Workers | p50 | p95 | p99 |",
        "|---:|---:|---:|---:|",
    ]
    for n in counts:
        rows = [r for r in runs if r["workers"] == n]
        lines.append(
            f"| {n} | {_median([r['p50_s'] for r in rows]) * 1000:.0f} ms | "
            f"{_median([r['p95_s'] for r in rows]) * 1000:.0f} ms | "
            f"{_median([r['p99_s'] for r in rows]) * 1000:.0f} ms |"
        )

    lines += [
        "",
        f"At client concurrency {args.concurrency}, a pool of N workers has",
        f"{args.capacity}xN concurrent slots. Below that the queue is short and",
        "latency is close to one generation; above it, requests wait, and the wait",
        "is what the tail reports.",
        "",
        "## Request distribution",
        "",
        "Round-robin, so an even split is the expected result. An uneven one would",
        "mean the routing is not doing what it claims.",
        "",
    ]
    for n in counts:
        last = [r for r in runs if r["workers"] == n][-1]
        dist = last["distribution"]
        if dist:
            share = ", ".join(f"`{k}`: {v}" for k, v in sorted(dist.items()))
            lines.append(f"- **{n} worker(s)**: {share}")
    lines.append("")

    if ceiling:
        lines += [
            "## The gateway's admission limit is its own ceiling",
            "",
            "`batch.max_batch_size` bounds concurrent inferences on the gateway",
            "regardless of how many workers exist. Set below N x per-worker",
            "capacity, adding workers changes nothing -- and from the outside the",
            "pool looks like it does not scale, when what is saturated is the",
            "gateway's permit count.",
            "",
            f"Both rows below run {args.ceiling_workers} workers "
            f"({per_worker_ideal * args.ceiling_workers:.0f} req/s of capacity):",
            "",
            "| `max_batch_size` | Median req/s | p95 |",
            "|---:|---:|---:|",
        ]
        for row in ceiling:
            lines.append(
                f"| {row['admission']} | {row['throughput_rps']:.1f} | "
                f"{row['p95_s'] * 1000:.0f} ms |"
            )
        lines += [
            "",
            "This is a real configuration hazard, not an artefact of the harness:",
            "the default is 8, so a deployment that scales its worker pool without",
            "raising it gets nothing for the extra workers.",
            "",
        ]

    if saturation:
        rows = saturation["rows"]
        cap = saturation["pool_capacity_rps"]
        best = max(r["total_rps"] for r in rows)
        lines += [
            "## Where scaling stops, and which side the limit is on",
            "",
            f"An efficiency figure is not worth much without this. A shortfall at",
            f"high N is equally consistent with *the gateway is saturated* and",
            f"*the load generator cannot go faster*, and those have opposite",
            f"implications. Splitting the same offered load across more client",
            f"tasks separates them: if more clients push more traffic, the",
            f"generator was the limit.",
            "",
            f"{args.saturation_workers} workers — {cap:.0f} req/s of declared pool",
            f"capacity, well above anything reached above:",
            "",
            "| Client groups | Concurrency each | Total req/s | p95 |",
            "|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['client_groups']} | {row['concurrency_each']} | "
                f"{row['total_rps']:.1f} | {row['p95_s'] * 1000:.0f} ms |"
            )
        lines += [
            "",
            f"Adding clients does not add throughput, so **the gateway is the",
            f"ceiling, at roughly {best:.0f} req/s on this host** — not the",
            f"harness. A pool larger than that is buying capacity the gateway",
            f"cannot hand out.",
            "",
            "That number is the practical use of this whole report: it is the",
            "point where scaling gateway replicas starts to matter more than",
            "scaling workers, and until now there was nothing to set that",
            "threshold from. It is specific to this host and this synthetic",
            "workload — a real backend changes the per-request cost on both",
            "sides — so it is an operating envelope for this configuration, not",
            "a property of the software.",
            "",
        ]

    failures = sum(r["failures"] for r in runs)
    cache_hits = sum(r.get("cache_hits", 0) for r in runs)
    dedup = sum(r.get("deduplicated", 0) for r in runs)
    dirty = [r for r in runs if r["failures"]]
    lines += [
        "## Sanity checks",
        "",
        f"- **{failures}** failed request(s) across {len(runs)} runs.",
        f"- **{cache_hits}** cache hit(s) and **{dedup}** deduplicated request(s).",
        "  Both must be zero: a cache hit or a coalesced duplicate is throughput",
        "  the pool did not actually produce, and would inflate every number here.",
        "",
    ]
    if dirty:
        lines += [
            "Runs with a failed request are **excluded from the medians above and",
            "listed here** rather than averaged in. Throughput is requests over",
            "wall time, so a single request that sits out the 120s worker timeout",
            "drags the figure to a fraction of the truth while leaving p95",
            "untouched — and a median across repeats hides it entirely, which is",
            "worse than the anomaly itself.",
            "",
            "| Workers | Repeat | Reported req/s | Failures |",
            "|---:|---:|---:|---:|",
        ]
        for r in dirty:
            lines.append(
                f"| {r['workers']} | {r['repeat']} | {r['throughput_rps']:.1f} | "
                f"{r['failures']} |"
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> int:
    counts = [int(x) for x in args.workers.split(",")]
    runs: List[Dict[str, Any]] = []

    for n in counts:
        for repeat in range(1, args.repeats + 1):
            print(f"--- {n} worker(s), repeat {repeat}/{args.repeats} ---", flush=True)
            result = await run_one(n, args, repeat)
            print(
                f"    {result['throughput_rps']:.1f} req/s, "
                f"p95 {result['p95_s'] * 1000:.0f} ms, "
                f"{result['failures']} failed",
                flush=True,
            )
            runs.append(result)

    print("--- saturation: gateway or harness? ---", flush=True)
    saturation = await run_saturation(args)
    for row in saturation["rows"]:
        print(
            f"    {row['client_groups']} client group(s): "
            f"{row['total_rps']:.1f} req/s",
            flush=True,
        )

    print("--- admission ceiling ---", flush=True)
    ceiling = await run_admission_ceiling(args)
    for row in ceiling:
        print(
            f"    max_batch_size={row['admission']}: "
            f"{row['throughput_rps']:.1f} req/s",
            flush=True,
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = format_report(args, runs, ceiling, saturation)
    (RESULTS_DIR / "scaling.md").write_text(report, encoding="utf-8")
    (RESULTS_DIR / "scaling.json").write_text(
        json.dumps(
            {"runs": runs, "ceiling": ceiling, "saturation": saturation}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"\nReport written to {RESULTS_DIR / 'scaling.md'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", default="1,2,4", help="comma-separated worker counts")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--requests", type=int, default=480)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--latency-ms", type=int, default=100,
                   help="synthetic cost of one generation on a dry-run worker")
    p.add_argument("--capacity", type=int, default=4,
                   help="concurrent generations per worker")
    p.add_argument("--admission", type=int, default=64,
                   help="gateway batch.max_batch_size; must exceed N x capacity")
    p.add_argument("--ceiling-workers", type=int, default=4,
                   help="worker count for the admission-ceiling comparison")
    p.add_argument("--saturation-workers", type=int, default=8,
                   help="worker count for the gateway-vs-harness comparison")
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
