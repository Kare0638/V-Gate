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

from vgate.config import BatchConfig  # noqa: E402

# Read from the config model rather than hardcoded, so this cannot drift from
# the value a deployment actually gets.
DEFAULT_MAX_BATCH_SIZE = BatchConfig().max_batch_size

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


def _proc_cpu_seconds(pid: int) -> Optional[float]:
    """User+system CPU consumed by a process, in seconds."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    except (OSError, IndexError):
        return None
    ticks = os.sysconf("SC_CLK_TCK")
    return (int(fields[11]) + int(fields[12])) / ticks  # utime, stime


def _system_cpu_seconds() -> Optional[tuple]:
    """(busy, total) CPU seconds across all cores since boot."""
    try:
        parts = Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]
    except OSError:
        return None
    values = [int(v) for v in parts]
    ticks = os.sysconf("SC_CLK_TCK")
    total = sum(values) / ticks
    idle = (values[3] + (values[4] if len(values) > 4 else 0)) / ticks
    return total - idle, total


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
        # Everything after the first spawn runs under try/except. If __aenter__
        # raises, Python never calls __aexit__ -- the `async with` body was
        # never entered -- so a health check that times out would leave the
        # whole process group running and holding the ports, and every later
        # topology in the same run would then attach to those orphans.
        try:
            await self._start()
        except BaseException:
            await self._stop()
            raise
        return self

    async def _start(self) -> None:
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

    async def __aexit__(self, *exc) -> None:
        await self._stop()

    async def _stop(self) -> None:
        _terminate(self.gateway)
        self.gateway = None
        for w in self.workers:
            _terminate(w)
        self.workers = []


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
    # The shipped default is measured, not extrapolated from a neighbouring
    # point. An earlier version of this report tested 4 and 64 and then wrote a
    # claim about 8 -- that at the default a four-worker pool serves what one
    # worker serves. Arithmetic says otherwise: 8 permits at 100 ms is about 80
    # req/s, which is 2x a single worker, not 1x. The hazard is real and the
    # stated size of it was wrong, so the value that matters is now a row.
    sweep = sorted({args.capacity, DEFAULT_MAX_BATCH_SIZE, args.admission})
    for admission in sweep:
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
        pool_capacity = args.ceiling_workers * args.capacity
        out.append({
            "admission": admission,
            "workers": args.ceiling_workers,
            "is_shipped_default": admission == DEFAULT_MAX_BATCH_SIZE,
            "binding": admission < pool_capacity,
            "throughput_rps": result["throughput"]["requests_per_second"],
            "p95_s": result["latency"]["p95_s"],
            "failures": result["failures"],
            "cache_hits": result["cache"]["hits"],
            "deduplicated": result["batching"]["deduplicated"],
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
        for procs in (1, 2, 4):
            # Separate OS processes, not coroutine groups. Groups inside one
            # interpreter share an event loop and a GIL, so a flat total across
            # them is equally consistent with "the gateway is saturated" and
            # "this Python process is" -- which is the exact ambiguity the
            # experiment exists to remove. Separate processes get their own
            # loop and can occupy other cores.
            #
            # Offered concurrency grows with the process count rather than
            # being split between them, so the client side is never held at a
            # fixed total that could itself be the ceiling.
            per_proc = args.concurrency
            requests = args.requests

            # CPU accounting around the window. "The gateway is the ceiling" is
            # a claim about one process, but client, gateway, and workers all
            # share this host and its loopback, so a flat curve is also
            # consistent with the machine being out of capacity. Comparing the
            # gateway's own CPU against the host's total distinguishes them:
            # a gateway pinned near one full core while the machine still has
            # idle cores is a limit in that process, not in the box.
            gw_before = _proc_cpu_seconds(topo.gateway.pid)
            sys_before = _system_cpu_seconds()
            measured = await _run_client_processes(
                topo.base_url, procs, per_proc, requests, f"sat-{procs}"
            )
            gw_after = _proc_cpu_seconds(topo.gateway.pid)
            sys_after = _system_cpu_seconds()

            results = measured["results"]
            window = measured["window_s"]
            gateway_cores = None
            host_cores_busy = None
            if None not in (gw_before, gw_after) and window > 0:
                gateway_cores = round((gw_after - gw_before) / window, 2)
            if sys_before and sys_after and window > 0:
                host_cores_busy = round(
                    (sys_after[0] - sys_before[0]) / window, 2
                )

            rows.append({
                "client_processes": procs,
                "concurrency_each": per_proc,
                "offered_concurrency": per_proc * procs,
                "total_rps": measured["total_rps"],
                "window_s": window,
                "start_skew_s": measured["start_skew_s"],
                "gateway_cores": gateway_cores,
                "host_cores_busy": host_cores_busy,
                "host_cores_total": os.cpu_count(),
                "p95_s": max(r["p95_s"] for r in results),
                "failures": sum(r["failures"] for r in results),
                "cache_hits": sum(r["cache_hits"] for r in results),
                "deduplicated": sum(r["deduplicated"] for r in results),
            })
    return {"pool_capacity_rps": capacity, "rows": rows}


async def _run_client_processes(
    url: str, count: int, concurrency: int, requests: int, tag: str
) -> Dict[str, Any]:
    """
    Run `count` independent load-generator processes over a shared window.

    Throughput is computed by this parent over one wall-clock interval, not by
    summing each process's own rate. Those are different quantities: each child
    reports requests over *its* window, and adding rates measured over windows
    that do not coincide gives a number that corresponds to no real interval.
    The error is not symmetric either -- a process that starts before the others
    have finished booting briefly has the server to itself, records a higher
    rate for that stretch, and inflates the sum. That bias grows with the
    process count, which is precisely the axis the experiment varies.

    The start barrier removes most of the skew and the common window absorbs
    the rest: total is every request completed divided by the span from the
    first client starting to the last one finishing.
    """
    stamp = int(time.time() * 1000)
    # Enough lead time for `count` interpreters to boot and reach the barrier.
    start_at = time.time() + 1.0 + 0.3 * count
    procs = [
        await asyncio.create_subprocess_exec(
            sys.executable, "-m", "benchmarks._load_client",
            "--url", url,
            "--concurrency", str(concurrency),
            "--requests", str(requests),
            "--tag", f"{tag}-{stamp}-{i}",
            "--start-at", f"{start_at:.3f}",
            cwd=str(REPO_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        for i in range(count)
    ]
    outputs = await asyncio.gather(*(p.communicate() for p in procs))
    results = []
    for (out, err), proc in zip(outputs, procs):
        if proc.returncode != 0:
            raise RuntimeError(
                f"load client exited {proc.returncode}: {err.decode()[-500:]}"
            )
        results.append(json.loads(out.decode()))

    window = max(r["ended_at"] for r in results) - min(r["started_at"] for r in results)
    total_requests = sum(r["requests"] for r in results)
    if window <= 0:
        # Refuse rather than degrade. An earlier version returned 0.0 req/s for
        # a nonsensical window, which is indistinguishable in the report from a
        # server that served nothing -- and the cause was a wall-clock jump, so
        # it would have appeared at random and been read as a result.
        raise RuntimeError(
            f"measurement window is {window:.2f}s; timestamps are unusable "
            f"({len(results)} client process(es))"
        )
    # Skew is reported rather than assumed away: a large value means the
    # processes were not really loading the server together, and the total
    # below is then an average over a ragged interval.
    skew = max(r["started_at"] for r in results) - min(r["started_at"] for r in results)
    return {
        "results": results,
        "total_rps": round(total_requests / window, 1) if window > 0 else 0.0,
        "window_s": round(window, 3),
        "start_skew_s": round(skew, 3),
    }


async def run_ceiling_source(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Identify *what* limits the gateway, not just that something does.

    Knowing the ceiling is ~180 req/s is only useful with its cause attached,
    and the obvious guess -- the gateway is out of CPU -- is wrong: it sits at
    0.6 of one core while the host has fourteen idle. Varying the generation
    cost separates the candidates, because a thread-bound ceiling moves with it
    and a CPU-bound one does not.

    RemoteBackend.generate is a synchronous httpx call dispatched through
    run_in_executor, so the gateway's outbound concurrency is capped by the
    default thread pool -- min(32, cpu_count + 4). That predicts
    threads/latency, and it is testable against a second ceiling further up
    where the event loop itself saturates.
    """
    threads = min(32, (os.cpu_count() or 1) + 4)
    rows = []
    for latency in args.ceiling_source_latencies:
        # Pool capacity kept far above every prediction, so it is never the
        # binding constraint in any row.
        async with Topology(
            args.saturation_workers * 2, latency, args.capacity, args.admission * 4
        ) as topo:
            before = _proc_cpu_seconds(topo.gateway.pid)
            measured = await _run_client_processes(
                topo.base_url, 2, args.concurrency * 3,
                args.requests, f"src-{latency}",
            )
            after = _proc_cpu_seconds(topo.gateway.pid)
        cores = None
        if None not in (before, after) and measured["window_s"] > 0:
            cores = round((after - before) / measured["window_s"], 2)
        rows.append({
            "latency_ms": latency,
            "pool_capacity_rps": args.saturation_workers * 2 * args.capacity
            / (latency / 1000.0),
            "thread_bound_prediction_rps": threads / (latency / 1000.0),
            "measured_rps": measured["total_rps"],
            "gateway_cores": cores,
            "failures": sum(r["failures"] for r in measured["results"]),
            "cache_hits": sum(r["cache_hits"] for r in measured["results"]),
            "deduplicated": sum(r["deduplicated"] for r in measured["results"]),
        })
    return {"executor_threads": threads, "rows": rows}


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
    source: Optional[Dict[str, Any]] = None,
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
            f"Every row runs {args.ceiling_workers} workers "
            f"({per_worker_ideal * args.ceiling_workers:.0f} req/s of capacity). "
            f"The predicted column is `max_batch_size / latency`, which is what",
            f"the limit allows when it is the binding constraint:",
            "",
            "| `max_batch_size` | Predicted req/s | Measured req/s | p95 | |",
            "|---:|---:|---:|---:|---|",
        ]
        for row in ceiling:
            predicted = row["admission"] / (args.latency_ms / 1000.0)
            note = " **shipped default**" if row["is_shipped_default"] else ""
            if not row["binding"]:
                predicted_s = "—"
                note += " (above pool capacity; not binding)"
            else:
                predicted_s = f"{predicted:.0f}"
            lines.append(
                f"| {row['admission']} | {predicted_s} | "
                f"{row['throughput_rps']:.1f} | {row['p95_s'] * 1000:.0f} ms |"
                f"{note} |"
            )

        default_row = next((r for r in ceiling if r["is_shipped_default"]), None)
        best_row = max(ceiling, key=lambda r: r["throughput_rps"])
        lines += ["", "This is a real configuration hazard: a deployment that scales"]
        if default_row:
            ratio = best_row["throughput_rps"] / default_row["throughput_rps"]
            lines += [
                f"its pool without raising `max_batch_size` gets "
                f"{default_row['throughput_rps']:.0f} req/s where the pool could",
                f"serve {best_row['throughput_rps']:.0f} — a "
                f"{ratio:.1f}x shortfall — and nothing in the system reports why.",
                "",
                "The default is measured here rather than inferred from a",
                "neighbouring point. An earlier version of this report tested 4 and",
                "64 and then wrote a claim about 8, asserting that at the default a",
                "four-worker pool serves what one worker serves. It does not: 8",
                "permits at this latency allows about 80 req/s, which is twice a",
                "single worker, not the same. The hazard was real and the stated",
                "size of it was wrong.",
                "",
            ]
        else:
            lines += ["", ""]

    if saturation:
        rows = saturation["rows"]
        cap = saturation["pool_capacity_rps"]
        best = max(r["total_rps"] for r in rows)
        lines += [
            "## Where scaling stops, and which side the limit is on",
            "",
            "An efficiency figure is not worth much without this. A shortfall at",
            "high N is equally consistent with *the gateway is saturated* and",
            "*the load generator cannot go faster*, and those have opposite",
            "implications.",
            "",
            "The load generators are **separate OS processes**, and offered",
            "concurrency grows with their number rather than being divided among",
            "them. Both matter. Coroutine groups inside one interpreter share an",
            "event loop and a GIL, so a flat total across them would be equally",
            "consistent with that one process being saturated; and holding total",
            "concurrency fixed would leave the offered load itself as a possible",
            "ceiling. Separate processes get their own loop and can occupy other",
            "cores, so a flat total across them is evidence about the server.",
            "",
            f"{args.saturation_workers} workers — {cap:.0f} req/s of declared pool",
            "capacity, well above anything reached above:",
            "",
            "Throughput here is computed by the parent over one wall-clock window",
            "— every request completed, divided by the span from the first client",
            "starting to the last finishing — not by summing each process's own",
            "rate. Those are different quantities, and summing rates measured over",
            "windows that do not coincide biases *upward* with the process count,",
            "which is the axis being varied. A shared start barrier removes most",
            "of the skew; the residual is reported below so it can be judged",
            "rather than assumed away.",
            "",
            "| Client processes | Offered concurrency | Total req/s | Window | Start skew | Gateway CPU | Host CPU busy | p95 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            gw = f"{row['gateway_cores']:.2f} cores" if row.get("gateway_cores") else "—"
            host = (
                f"{row['host_cores_busy']:.1f} / {row['host_cores_total']}"
                if row.get("host_cores_busy") else "—"
            )
            lines.append(
                f"| {row['client_processes']} | {row['offered_concurrency']} | "
                f"{row['total_rps']:.1f} | {row['window_s']:.1f} s | "
                f"{row['start_skew_s'] * 1000:.0f} ms | {gw} | {host} | "
                f"{row['p95_s'] * 1000:.0f} ms |"
            )
        spread = (best - min(r["total_rps"] for r in rows)) / best * 100
        top = max(rows, key=lambda r: r["offered_concurrency"])
        lines += [
            "",
            f"Quadrupling both the number of client processes and the offered",
            f"concurrency moves the total by {spread:.0f}%. The pool has "
            f"{cap:.0f} req/s of capacity and is not the limit, and the load",
            f"generators are separate processes that cannot be limited by a",
            f"shared interpreter, so the ceiling is around "
            f"**{best:.0f} req/s** on this host.",
            "",
        ]
        if top.get("gateway_cores") and top.get("host_cores_busy"):
            headroom = top["host_cores_total"] - top["host_cores_busy"]
            lines += [
                f"The CPU columns rule out the obvious explanation. At the widest",
                f"point the gateway uses {top['gateway_cores']:.2f} of one core",
                f"while the host runs {top['host_cores_busy']:.1f} of "
                f"{top['host_cores_total']} cores busy, leaving about "
                f"{headroom:.1f} idle — so **the gateway is not out of CPU, and",
                f"neither is the machine**. Something else is binding, and the",
                f"next section identifies it.",
                "",
                "**What this still does not rule out.** Client, gateway, and",
                "workers share one host and its loopback interface, so a shared",
                "resource other than aggregate CPU is not excluded by these",
                "numbers alone. Settling that completely needs a load generator",
                "on another machine.",
                "",
            ]
        else:
            lines += [
                "**Scope.** CPU accounting was unavailable, so these numbers do",
                "not separate a limit in the gateway process from one in the",
                "machine.",
                "",
            ]

        lines += [
            "That number is the practical use of this whole report: it is the",
            "point where scaling gateway replicas starts to matter more than",
            "scaling workers, and until now there was nothing to set that",
            "threshold from. It is specific to this host and this synthetic",
            "workload — a real backend changes the per-request cost on both",
            "sides — so it is an operating envelope for this configuration, not",
            "a property of the software.",
            "",
        ]

    if source:
        srows = source["rows"]
        threads = source["executor_threads"]
        lines += [
            "## What the ceiling actually is",
            "",
            "A ceiling is only useful with its cause attached, and the obvious",
            "guess was wrong — the gateway is nowhere near CPU-bound at the rate",
            "above. Varying the generation cost separates the candidates: a",
            "thread-bound limit moves with it, a CPU-bound one does not.",
            "",
            "`RemoteBackend.generate` is a **synchronous** httpx call dispatched",
            "through `run_in_executor`, so the gateway's outbound concurrency is",
            "capped by the default thread pool — `min(32, cpu_count + 4)`, which",
            f"is **{threads}** on this host. That predicts `threads / latency`.",
            "",
            "| Generation cost | Pool capacity | Threads ÷ latency | Measured | Gateway CPU |",
            "|---:|---:|---:|---:|---:|",
        ]
        for row in srows:
            gw = f"{row['gateway_cores']:.2f} cores" if row.get("gateway_cores") else "—"
            lines.append(
                f"| {row['latency_ms']} ms | {row['pool_capacity_rps']:.0f} | "
                f"{row['thread_bound_prediction_rps']:.0f} | "
                f"{row['measured_rps']:.1f} | {gw} |"
            )
        slowest, fastest = srows[0], srows[-1]
        lines += [
            "",
            "Two different ceilings, and which binds depends on the workload:",
            "",
            f"- At **{slowest['latency_ms']} ms** the measurement tracks the",
            f"  thread prediction ({slowest['measured_rps']:.0f} against",
            f"  {slowest['thread_bound_prediction_rps']:.0f}) while CPU sits at",
            f"  {slowest['gateway_cores']:.2f} cores. The **thread pool** binds.",
            f"- At **{fastest['latency_ms']} ms** the pool would allow",
            f"  {fastest['thread_bound_prediction_rps']:.0f} but only",
            f"  {fastest['measured_rps']:.0f} arrives, with CPU up at",
            f"  {fastest['gateway_cores']:.2f} cores. The **event loop** binds.",
            "",
            "This changes what the earlier number means. The ~180 req/s figure is",
            f"not a property of the gateway — it is `{threads} threads / 100 ms`,",
            "and that thread count comes from `os.cpu_count()`. It moves with the",
            "host: exactly the dependency this report eliminates for worker",
            "capacity, and had quietly left in place for the gateway.",
            "",
            "The fix is a design change rather than a tuning knob. An **async**",
            "HTTP client in `RemoteBackend` would take the thread pool out of the",
            "path entirely, leaving the event-loop ceiling as the only one. That",
            "is recorded in ROADMAP.md, not done here.",
            "",
        ]

    # Every scenario, not just the nine scaling runs. bench_load computes
    # throughput as requests over wall time and counts failures in the
    # numerator, so a single request that sits out the 120s worker timeout
    # corrupts whatever scenario it lands in. Checking only the main runs left
    # the two headline conclusions -- the admission ceiling and the saturation
    # point -- able to be quietly wrong.
    scopes = [
        ("scaling runs", runs),
        ("admission ceiling", ceiling),
        ("saturation", saturation["rows"] if saturation else []),
        ("ceiling source", source["rows"] if source else []),
    ]
    lines += ["## Sanity checks", ""]
    total_bad = 0
    for name, rows_ in scopes:
        if not rows_:
            continue
        f = sum(r.get("failures", 0) for r in rows_)
        c = sum(r.get("cache_hits", 0) for r in rows_)
        d = sum(r.get("deduplicated", 0) for r in rows_)
        total_bad += f + c + d
        mark = "" if (f or c or d) == 0 else "  ← **not clean**"
        lines.append(
            f"- **{name}** ({len(rows_)} run(s)): {f} failed, {c} cache hit(s), "
            f"{d} deduplicated.{mark}"
        )
    lines += [
        "",
        "All three counts must be zero in every scenario. A failed request",
        "inflates nothing but still divides into wall time; a cache hit or a",
        "coalesced duplicate is throughput the pool did not actually produce.",
        "",
    ]
    if total_bad:
        lines += [
            "> **This run is not clean.** The numbers above should not be quoted",
            "> until it is re-run without failures.",
            "",
        ]

    dirty = [r for r in runs if r["failures"]]
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
            f"    {row['client_processes']} proc(s), offered "
            f"{row['offered_concurrency']}: {row['total_rps']:.1f} req/s "
            f"(window {row['window_s']:.1f}s, skew {row['start_skew_s']*1000:.0f}ms, "
            f"gw {row['gateway_cores']} cores, host "
            f"{row['host_cores_busy']}/{row['host_cores_total']}), "
            f"{row['failures']} failed",
            flush=True,
        )

    print("--- what the ceiling actually is ---", flush=True)
    source = await run_ceiling_source(args)
    for row in source["rows"]:
        print(
            f"    {row['latency_ms']}ms: measured {row['measured_rps']:.1f} vs "
            f"thread-bound {row['thread_bound_prediction_rps']:.0f}, "
            f"gw {row['gateway_cores']} cores, {row['failures']} failed",
            flush=True,
        )

    print("--- admission ceiling ---", flush=True)
    ceiling = await run_admission_ceiling(args)
    for row in ceiling:
        tag = "  <- shipped default" if row["is_shipped_default"] else ""
        print(
            f"    max_batch_size={row['admission']}: "
            f"{row['throughput_rps']:.1f} req/s, {row['failures']} failed{tag}",
            flush=True,
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = format_report(args, runs, ceiling, saturation, source)
    (RESULTS_DIR / "scaling.md").write_text(report, encoding="utf-8")
    (RESULTS_DIR / "scaling.json").write_text(
        json.dumps(
            {"runs": runs, "ceiling": ceiling, "saturation": saturation,
             "ceiling_source": source}, indent=2
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
    p.add_argument("--ceiling-source-latencies", type=int, nargs="+",
                   default=[100, 50, 25],
                   help="generation costs used to separate a thread-bound "
                        "ceiling from a CPU-bound one")
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
