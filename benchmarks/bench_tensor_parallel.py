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
Measure what a second GPU buys, by running the same model at TP=1 and TP=2.

WHY THE MODEL MUST FIT ON ONE GPU

The obvious choice is a model too large for one device, so tensor parallelism
is mandatory. That choice makes the measurement impossible: with no TP=1 run
there is no baseline, and the result degrades to "it ran". Pick a model that
fits on one device but leaves it short of KV cache, and both configurations
are measurable on identical work.

WHAT TO EXPECT, AND WHY BOTH HALVES MATTER

Sharding weights frees memory per device, and that memory becomes KV cache,
which is what bounds concurrency once a model is loaded. So TP=2 should serve
more concurrent work.

It should also make a *single* request slightly slower: every layer ends in an
all-reduce, on the critical path of every token. A report that shows only
throughput is hiding half of the trade, so this sweeps concurrency and reports
latency at each point. If TP=2 wins everywhere including concurrency 1,
something is wrong with the measurement, not with the hardware.

HONESTY REQUIREMENTS BUILT IN

- The environment is captured, not described: driver, vLLM version, GPU model,
  and `nvidia-smi topo -m`. A TP number without the interconnect is
  uninterpretable -- NVLink and PCIe differ by roughly 10x -- and unreproducible.
- KV cache blocks are read from vLLM's own startup log for each configuration,
  because "TP=2 was faster" is a result and "TP=2 had 4x the KV cache" is the
  reason.
- Every run asserts: no failed requests, no cache hits, no deduplication. Any
  of those means throughput was reported for work the GPUs did not do.
- The client is checked against being the bottleneck, since a flat curve is
  otherwise equally consistent with the server being saturated.

SELF-TEST FIRST

    python benchmarks/bench_tensor_parallel.py --self-test

runs the entire harness against the dry-run backend on CPU. Nothing it
measures is meaningful, but every code path executes -- spawn, load, sweep,
assertions, report. Harness bugs found on a rented A100 are billed by the hour;
find them here first.

Usage:
    python benchmarks/bench_tensor_parallel.py --self-test
    python benchmarks/bench_tensor_parallel.py --model Qwen/Qwen2.5-32B-Instruct
"""

import argparse
import asyncio
import json
import os
import re
import shutil
import signal
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
PORT = 8120
BASE_URL = f"http://127.0.0.1:{PORT}"


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------

def _run(cmd: List[str]) -> Optional[str]:
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip() or None


def capture_environment() -> Dict[str, Any]:
    """
    Everything needed to interpret or reproduce the numbers.

    The topology matrix is the load-bearing part. Tensor parallelism does an
    all-reduce per layer, so its cost is set by the link between the devices,
    and NVLink versus PCIe is roughly a tenfold difference. A TP result without
    it cannot be compared to anything.
    """
    env: Dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "topology": _run(["nvidia-smi", "topo", "-m"]),
        "gpus": _run([
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ]),
        "cpu_count": os.cpu_count(),
    }
    # Version probes run in a subprocess rather than importing here. Importing
    # torch and vLLM into the harness loads enough state to slow its own event
    # loop, which used to matter when this process also generated load; it no
    # longer does, and keeping the import out removes the possibility of that
    # coupling returning unnoticed.
    probe = _run([
        sys.executable, "-c",
        "import json;"
        "d={};"
        "\ntry:\n import vllm; d['vllm']=vllm.__version__\nexcept Exception: d['vllm']=None"
        "\ntry:\n import torch; d['torch']=torch.__version__; d['cuda']=torch.version.cuda"
        "\nexcept Exception: d['torch']=d['cuda']=None"
        "\nprint(json.dumps(d))",
    ])
    try:
        versions = json.loads(probe) if probe else {}
    except json.JSONDecodeError:
        versions = {}
    env["vllm_version"] = versions.get("vllm")
    env["torch_version"] = versions.get("torch")
    env["cuda_version"] = versions.get("cuda")
    return env


def interconnect_of(topology: Optional[str]) -> Optional[str]:
    """
    The link between GPU0 and GPU1, read from `nvidia-smi topo -m`.

    NV# is NVLink; PIX/PXB stay inside a PCIe switch; PHB crosses a host
    bridge; SYS crosses sockets and is the worst case. The report prints this
    rather than letting a reader assume the best one.
    """
    if not topology:
        return None
    for line in topology.splitlines():
        if line.startswith("GPU0"):
            fields = line.split()
            if len(fields) >= 3:
                return fields[2]
    return None


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def _fp16_config_path() -> Path:
    """
    A config file whose `quantization` is a real null.

    Needed because the environment cannot express one: VGATE_MODEL__QUANTIZATION
    is an Optional[str], and setting it to "null" yields the four-character
    string, which vLLM then tries to look up as a quantization method. Written
    once and reused; the YAML source is lower priority than the environment, so
    every other override in _spawn still wins.
    """
    path = RESULTS_DIR / "logs" / "fp16-config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("model:\n  quantization: null\n", encoding="utf-8")
    return path


def _spawn(env_overrides: Dict[str, str], log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env.update({
        "VGATE_SERVER__PORT": str(PORT),
        "VGATE_SECURITY__ENABLED": "false",
        "VGATE_TRACING__ENABLED": "false",
        "VGATE_LOGGING__LEVEL": "WARNING",
        # The cache would answer repeated prompts without touching a GPU. All
        # prompts here are unique, so this is belt and braces -- but a silent
        # cache hit is throughput credited to hardware that did nothing.
        "VGATE_CACHE__ENABLED": "false",
        # vLLM crashes at startup under WSL2 without this. Carried over from
        # run_report.py, where it was found the hard way; harmless elsewhere.
        "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
    })
    env.update(env_overrides)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=open(log_path, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        # vLLM spawns EngineCore workers; signalling only main.py leaves them
        # holding GPU memory, and the next configuration then fails to load.
        start_new_session=True,
    )


def _terminate(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 30)):
        try:
            os.killpg(pgid, sig)
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue
        except ProcessLookupError:
            return


async def _wait_healthy(timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            try:
                async with session.get(
                    f"{BASE_URL}/health", timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                pass
            await asyncio.sleep(1.0)
    raise RuntimeError(f"server did not become healthy within {timeout_s}s")


def read_kv_cache_blocks(log_path: Path) -> Optional[int]:
    """
    GPU KV cache blocks, from vLLM's own startup line.

    This is the mechanism behind whatever the throughput numbers show. Sharding
    the weights frees device memory, that memory becomes KV cache, and KV cache
    is what bounds how many sequences can run at once. Reporting the result
    without it leaves the reader to guess why.
    """
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # vLLM has phrased this differently across versions; accept the variants.
    for pattern in (
        r"#\s*GPU blocks:\s*([\d,]+)",
        r"GPU KV cache size:\s*([\d,]+)",
        r"gpu_blocks[\"']?[:=]\s*([\d,]+)",
    ):
        m = re.search(pattern, text)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

class Failure(Exception):
    """A condition that makes the numbers not worth reporting."""


# Prompt generation lives in _load_client now that measurement does. Prompts
# are distinct there for the same reason it mattered here: a cache hit or a
# coalesced duplicate is throughput credited to a GPU that did nothing.


async def measure(
    concurrency: int, requests: int, max_tokens: int, tag: str
) -> Dict[str, Any]:
    """
    Drive load from a clean subprocess, never from this one.

    Measuring in-process looked simpler and was wrong by about 40%. This
    harness imports torch and vLLM to report their versions, and an event loop
    sharing an interpreter with that much loaded state cannot drive requests as
    fast as a fresh one: on an RTX 3060, in-process reported 19 req/s where a
    subprocess at identical concurrency reported 33 against the same server.
    Every figure in the sweep would have been depressed, and not necessarily by
    the same amount at TP=1 and TP=2 -- which would have corrupted the
    comparison rather than merely the absolute numbers.
    """
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "benchmarks._load_client",
        "--url", BASE_URL,
        "--concurrency", str(concurrency),
        "--requests", str(requests),
        "--max-tokens", str(max_tokens),
        "--tag", f"{tag}-{int(time.time() * 1000)}",
        cwd=str(REPO_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise Failure(f"load client exited {proc.returncode}: {err.decode()[-400:]}")
    result = json.loads(out.decode())

    row = {
        "concurrency": concurrency,
        "requests": requests,
        "rps": result["requests_per_second"],
        "tokens_per_second": result["tokens_per_second"],
        "p50_s": result["p50_s"],
        "p95_s": result["p95_s"],
        "p99_s": result["p99_s"],
        "failures": result["failures"],
        "cache_hits": result["cache_hits"],
        "deduplicated": result["deduplicated"],
    }
    # Asserted here, not tallied at the end. A run with any of these did not
    # measure what the report will claim it measured, and continuing would put
    # a number in a table that has to be walked back later.
    if row["failures"]:
        raise Failure(f"{row['failures']} request(s) failed at concurrency {concurrency}")
    if row["cache_hits"]:
        raise Failure(
            f"{row['cache_hits']} cache hit(s) at concurrency {concurrency}: "
            "throughput would include work no GPU performed"
        )
    if row["deduplicated"]:
        raise Failure(
            f"{row['deduplicated']} deduplicated request(s) at concurrency "
            f"{concurrency}: prompts were not unique"
        )
    return row


async def run_configuration(
    tp: int, args: argparse.Namespace
) -> Dict[str, Any]:
    """Load the model at this tensor-parallel size and sweep concurrency."""
    log_path = RESULTS_DIR / "logs" / f"tp{tp}-server.log"
    overrides = {
        "VGATE_MODEL__TENSOR_PARALLEL_SIZE": str(tp),
        "VGATE_MODEL__MODEL_ID": args.model,
        "VGATE_MODEL__MAX_MODEL_LEN": str(args.max_model_len),
        "VGATE_MODEL__GPU_MEMORY_UTILIZATION": str(args.gpu_memory_utilization),
        # CUDA graphs on: eager mode is a debugging aid and measuring with it
        # reports a number no deployment would see.
        "VGATE_MODEL__ENFORCE_EAGER": "false",
        # Admission must not be what limits concurrency here -- the question is
        # what the GPUs can do, and max_batch_size would silently cap it.
        "VGATE_BATCH__MAX_BATCH_SIZE": str(max(args.concurrency) * 2),
        "VGATE_RELIABILITY__REQUEST_TIMEOUT_SECONDS": str(args.request_timeout),
    }
    if args.self_test:
        overrides["VGATE_DRY_RUN"] = "true"
        overrides["VGATE_DRYRUN_SIMULATED_LATENCY_MS"] = "50"
    else:
        overrides["VGATE_MODEL__ENGINE_TYPE"] = "vllm"
        if args.quantization:
            overrides["VGATE_MODEL__QUANTIZATION"] = args.quantization
        else:
            # Clearing an Optional[str] through the environment does not work:
            # pydantic-settings hands the string "null" straight through, and
            # vLLM then looks for a quantization method by that name and fails
            # at load. Every FP16 checkpoint would have hit this. The YAML
            # source is what can express a real null, so this points the
            # process at a generated config instead.
            overrides["VGATE_CONFIG_PATH"] = str(_fp16_config_path())

    proc = _spawn(overrides, log_path)
    try:
        await _wait_healthy(args.load_timeout)
        kv_blocks = read_kv_cache_blocks(log_path)

        # Warmup is discarded. The first requests pay for CUDA graph capture
        # and allocator growth, which belong to startup rather than to steady
        # state, and folding them in would flatter whichever configuration ran
        # second.
        await measure(
            concurrency=min(4, max(args.concurrency)),
            requests=args.warmup_requests,
            max_tokens=args.max_tokens,
            tag=f"warmup-tp{tp}",
        )

        rows: List[Dict[str, Any]] = []
        for concurrency in args.concurrency:
            for repeat in range(1, args.repeats + 1):
                row = await measure(
                    concurrency=concurrency,
                    requests=args.requests,
                            max_tokens=args.max_tokens,
                    tag=f"tp{tp}-c{concurrency}-r{repeat}",
                )
                row["repeat"] = repeat
                rows.append(row)
                print(
                    f"    TP={tp} c={concurrency} r={repeat}: "
                    f"{row['rps']:.2f} req/s, {row['tokens_per_second']:.0f} tok/s, "
                    f"p95 {row['p95_s'] * 1000:.0f} ms",
                    flush=True,
                )
    finally:
        _terminate(proc)
        # vLLM releases GPU memory on exit, but not instantly; loading the next
        # configuration into memory the previous one still holds fails in a way
        # that looks like the model is too large.
        await asyncio.sleep(args.settle_seconds)

    return {"tensor_parallel_size": tp, "kv_cache_blocks": kv_blocks, "rows": rows}


async def check_client_headroom(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Confirm the load generator is not what the numbers are measuring.

    A curve that flattens is equally consistent with the server being saturated
    and the client being unable to push harder, and those have opposite
    meanings. Splitting the same offered load across independent processes
    separates them: if more processes move more traffic, the generator was the
    limit and every throughput figure above is a floor rather than a result.
    """
    top = max(args.concurrency)
    # Deliberately more requests than a sweep point uses. A short window is
    # dominated by ramp-up, and the comparison then reports the shape of the
    # first second rather than a steady rate.
    args = argparse.Namespace(**{**vars(args), "requests": max(args.requests, 256)})
    single = await measure(
        concurrency=top, requests=args.requests,
        max_tokens=args.max_tokens,
        tag="headroom-1proc",
    )
    stamp = int(time.time() * 1000)
    # A shared wall-clock start. Without it the processes stagger by however
    # long an interpreter takes to boot, one of them briefly has the server to
    # itself, and the aggregate over a common window is inflated -- which would
    # manufacture exactly the "the client was the limit" verdict this is
    # supposed to test for. The same omission was already fixed once in
    # bench_scaling.py and repeated here.
    start_at = time.time() + 2.0
    procs = [
        await asyncio.create_subprocess_exec(
            sys.executable, "-m", "benchmarks._load_client",
            "--url", BASE_URL,
            "--concurrency", str(top // 2),
            "--requests", str(args.requests // 2),
            "--tag", f"headroom-2proc-{stamp}-{i}",
            "--max-tokens", str(args.max_tokens),
            "--start-at", f"{start_at:.3f}",
            cwd=str(REPO_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        for i in range(2)
    ]
    outputs = await asyncio.gather(*(p.communicate() for p in procs))
    results = []
    for (out, err), proc in zip(outputs, procs):
        if proc.returncode != 0:
            raise Failure(f"load client exited {proc.returncode}: {err.decode()[-400:]}")
        results.append(json.loads(out.decode()))

    window = max(r["ended_at"] for r in results) - min(r["started_at"] for r in results)
    if window <= 0:
        raise Failure(f"client window is {window:.2f}s; timestamps unusable")
    split_rps = sum(r["requests"] for r in results) / window
    return {
        "single_process_rps": single["rps"],
        "two_process_rps": round(split_rps, 2),
        "gain_pct": round(100 * (split_rps - single["rps"]) / single["rps"], 1),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _median(rows: List[Dict[str, Any]], key: str) -> float:
    return statistics.median([r[key] for r in rows])


def _at(config: Dict[str, Any], concurrency: int) -> List[Dict[str, Any]]:
    return [r for r in config["rows"] if r["concurrency"] == concurrency]


def format_report(
    args: argparse.Namespace,
    env: Dict[str, Any],
    configs: List[Dict[str, Any]],
    headroom: Dict[str, Any],
) -> str:
    link = interconnect_of(env.get("topology"))
    by_tp = {c["tensor_parallel_size"]: c for c in configs}
    tps = sorted(by_tp)

    lines = [
        "# Tensor Parallelism: one GPU versus two",
        "",
        f"Generated by `benchmarks/bench_tensor_parallel.py` on "
        f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC.",
        "Regenerate with that script; do not edit by hand.",
        "",
    ]

    if args.self_test:
        lines += [
            "> **SELF-TEST RUN — the numbers below are meaningless.**",
            "> This exercised the harness against the dry-run backend on CPU to",
            "> shake out bugs before paying for GPU time. No GPU was involved and",
            "> nothing here describes tensor parallelism.",
            "",
        ]

    if len(tps) == 1:
        lines += [
            f"> **Single-configuration run (TP={tps[0]} only).** This is not a",
            "> tensor-parallelism comparison — there is nothing to compare against.",
            "> It exercises the harness against a real GPU and a real engine, which",
            "> is what the `--self-test` path on CPU cannot do.",
            "",
        ]

    lines += [
        "## What was measured, and on what",
        "",
        "The same model at TP=1 and TP=2, over the same work. A model that only",
        "fits on two devices would have made this impossible: without a TP=1 run",
        "there is no baseline and the result degrades to \"it ran\".",
        "",
        "| | |",
        "|---|---|",
        f"| Model | `{args.model}` |" if not args.self_test
        else "| Model | *none — dry-run backend, no model was loaded* |",
        f"| Quantization | {args.quantization or 'none (native precision)'} |"
        if not args.self_test else "| Quantization | *n/a* |",
        f"| Max model length | {args.max_model_len} |",
        f"| GPU memory utilization | {args.gpu_memory_utilization} |",
        "| Prompts | all distinct, so no cache hit or coalesced duplicate |",
        f"| Generated tokens per request | {args.max_tokens} |",
        f"| Requests per point | {args.requests} |",
        f"| Repeats per point | {args.repeats} |",
        f"| GPUs | {env.get('gpus') or 'unknown'} |",
        f"| **GPU0↔GPU1 link** | **{link or 'unknown'}** |",
        f"| vLLM | {env.get('vllm_version') or 'unknown'} |",
        f"| torch / CUDA | {env.get('torch_version') or '?'} / {env.get('cuda_version') or '?'} |",
        "",
        "The interconnect row is not decoration. Tensor parallelism ends every",
        "layer with an all-reduce, on the critical path of every token, so its",
        "cost is set by the link between the devices — and NVLink versus PCIe is",
        "roughly a tenfold difference in bandwidth. A TP result quoted without it",
        "cannot be compared to anything, including a rerun on another host.",
        "",
        "```",
        (env.get("topology") or "nvidia-smi topo -m unavailable").strip(),
        "```",
        "",
    ]

    # KV cache is the mechanism; state it before the results it explains.
    blocks = {tp: by_tp[tp].get("kv_cache_blocks") for tp in tps}
    if all(blocks.get(tp) for tp in tps):
        lines += [
            "## Why the numbers come out the way they do",
            "",
            "| TP | GPU KV cache blocks | vs TP=1 |",
            "|---:|---:|---:|",
        ]
        base_blocks = blocks[tps[0]]
        for tp in tps:
            lines.append(
                f"| {tp} | {blocks[tp]:,} | {blocks[tp] / base_blocks:.2f}x |"
            )
        lines += [
            "",
            "Sharding the weights frees device memory, and that memory becomes KV",
            "cache — which is what bounds how many sequences can run at once after",
            "a model is loaded. This ratio, not raw compute, is the mechanism",
            "behind most of the throughput difference below.",
            "",
        ]

    lines += [
        "## Throughput",
        "",
        "| Concurrency | " + " | ".join(f"TP={tp} req/s" for tp in tps) + " | Speedup |",
        "|---:|" + "---:|" * (len(tps) + 1),
    ]
    for c in args.concurrency:
        cells, base = [], None
        for tp in tps:
            rows = _at(by_tp[tp], c)
            med = _median(rows, "rps") if rows else float("nan")
            if base is None:
                base = med
            cells.append(f"{med:.2f}")
        speedup = f"{(float(cells[-1]) / base):.2f}x" if base else "—"
        lines.append(f"| {c} | " + " | ".join(cells) + f" | {speedup} |")

    lines += [
        "",
        "## Latency, which is the other half of the trade",
        "",
        "| Concurrency | " + " | ".join(f"TP={tp} p50 / p95" for tp in tps) + " |",
        "|---:|" + "---:|" * len(tps),
    ]
    for c in args.concurrency:
        cells = []
        for tp in tps:
            rows = _at(by_tp[tp], c)
            if rows:
                cells.append(
                    f"{_median(rows, 'p50_s') * 1000:.0f} / "
                    f"{_median(rows, 'p95_s') * 1000:.0f} ms"
                )
            else:
                cells.append("—")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")

    # State the expected shape, then whether it held. Saying it afterwards
    # would be indistinguishable from fitting the story to the data.
    low, high = args.concurrency[0], args.concurrency[-1]
    if len(tps) >= 2:
        base_tp, top_tp = tps[0], tps[-1]
        low_rows_b, low_rows_t = _at(by_tp[base_tp], low), _at(by_tp[top_tp], low)
        if low_rows_b and low_rows_t:
            lat_b = _median(low_rows_b, "p50_s") * 1000
            lat_t = _median(low_rows_t, "p50_s") * 1000
            got_slower = lat_t > lat_b
            lines += [
                "",
                f"At concurrency {low} the expected result is that TP={top_tp} is",
                f"**slower** than TP={base_tp}: one request cannot use the extra",
                "device for anything, and it now pays an all-reduce per layer.",
                f"Measured: {lat_b:.0f} ms against {lat_t:.0f} ms — "
                + ("as expected." if got_slower else
                   "**not** as expected, which is worth explaining before the "
                   "throughput figures are trusted, since a configuration that "
                   "wins everywhere usually means the two runs were not doing "
                   "equal work."),
                "",
                "Tensor parallelism is a throughput-for-latency trade. A report",
                "showing only the throughput side is showing half a result.",
                "",
            ]

    lines += [
        "## Is the client the bottleneck?",
        "",
        "A curve that flattens is equally consistent with the server being",
        "saturated and the load generator being unable to push harder, and those",
        "mean opposite things. Splitting the same offered load across two",
        "independent processes separates them.",
        "",
        f"- one process: **{headroom['single_process_rps']:.2f} req/s**",
        f"- two processes: **{headroom['two_process_rps']:.2f} req/s** "
        f"({headroom['gain_pct']:+.1f}%)",
        "",
    ]
    if headroom["gain_pct"] > 10:
        lines += [
            "> **The generator was a limit.** Two processes moved materially more",
            "> traffic than one, so the throughput figures above are floors, not",
            "> measurements of what the GPUs can do. Rerun with more client",
            "> processes before quoting them.",
            "",
        ]
    else:
        lines += [
            "More client processes do not move more traffic, so the figures above",
            "are bounded by the server rather than by the harness.",
            "",
        ]

    lines += [
        "## What this does not establish",
        "",
        f"- One model at one length on one host. TP behaviour depends on model",
        "  architecture, sequence length, and the interconnect above; none of",
        "  these numbers transfer to a different combination.",
        "- No comparison against a different interconnect. Whether these results",
        "  would hold on PCIe instead of NVLink is not measured here, and the",
        "  difference is large enough that it must not be assumed.",
        "- Throughput is measured at the gateway, so it includes the gateway's",
        "  own per-request cost. That cost was separately measured at roughly",
        "  180 req/s on a different host ([scaling.md](scaling.md)); at the rates",
        "  reached here it is not the binding constraint, but it is not zero.",
        "",
        "## Sanity checks",
        "",
        "Every run asserted zero failed requests, zero cache hits, and zero",
        "deduplicated requests, and aborted rather than reporting if any were",
        "non-zero. A cache hit is throughput credited to a GPU that did nothing;",
        "a failure still divides into wall time. Neither can be discovered after",
        "the fact from a table of medians.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    env = capture_environment()
    link = interconnect_of(env.get("topology"))
    print(f"GPUs:         {env.get('gpus') or 'none detected'}")
    print(f"GPU0<->GPU1:  {link or 'unknown'}")
    print(f"vLLM:         {env.get('vllm_version') or 'not installed'}")

    if not args.self_test:
        if not env.get("gpus"):
            raise Failure("no GPUs detected; use --self-test to exercise the harness")
        # SYS crosses sockets and PHB crosses a host bridge; either makes the
        # all-reduce dominate, and the result then describes the machine's
        # wiring rather than tensor parallelism. Refuse by default rather than
        # publish a number whose caveat nobody will read.
        if (
            max(args.tensor_parallel) > 1
            and link in (None, "SYS", "PHB")
            and not args.allow_slow_interconnect
        ):
            raise Failure(
                f"GPU0<->GPU1 link is {link!r}. Tensor parallelism over this "
                "measures the interconnect, not the GPUs. Re-run with "
                "--allow-slow-interconnect to proceed deliberately; the report "
                "will carry the link either way."
            )

    configs = []
    for tp in args.tensor_parallel:
        print(f"\n--- tensor_parallel_size={tp} ---", flush=True)
        configs.append(await run_configuration(tp, args))

    # Headroom is checked against the last configuration, which is still the
    # one that reached the highest rates and is therefore where a client limit
    # would show first.
    print("\n--- client headroom ---", flush=True)
    proc = _spawn(
        {
            "VGATE_MODEL__TENSOR_PARALLEL_SIZE": str(args.tensor_parallel[-1]),
            "VGATE_MODEL__MODEL_ID": args.model,
            "VGATE_MODEL__MAX_MODEL_LEN": str(args.max_model_len),
            "VGATE_MODEL__GPU_MEMORY_UTILIZATION": str(args.gpu_memory_utilization),
            "VGATE_MODEL__ENFORCE_EAGER": "false",
            "VGATE_BATCH__MAX_BATCH_SIZE": str(max(args.concurrency) * 2),
            "VGATE_RELIABILITY__REQUEST_TIMEOUT_SECONDS": str(args.request_timeout),
            **({"VGATE_DRY_RUN": "true", "VGATE_DRYRUN_SIMULATED_LATENCY_MS": "50"}
               if args.self_test else
               {"VGATE_MODEL__ENGINE_TYPE": "vllm",
                **({"VGATE_MODEL__QUANTIZATION": args.quantization}
                   if args.quantization
                   else {"VGATE_CONFIG_PATH": str(_fp16_config_path())})}),
        },
        RESULTS_DIR / "logs" / "tp-headroom-server.log",
    )
    try:
        await _wait_healthy(args.load_timeout)
        headroom = await check_client_headroom(args)
        print(
            f"    1 proc {headroom['single_process_rps']:.2f} req/s vs "
            f"2 proc {headroom['two_process_rps']:.2f} req/s "
            f"({headroom['gain_pct']:+.1f}%)",
            flush=True,
        )
    finally:
        _terminate(proc)

    name = "tensor_parallel_selftest" if args.self_test else "tensor_parallel"
    (RESULTS_DIR / f"{name}.md").write_text(
        format_report(args, env, configs, headroom), encoding="utf-8"
    )
    (RESULTS_DIR / f"{name}.json").write_text(
        json.dumps(
            {"args": vars(args), "environment": env, "configs": configs,
             "client_headroom": headroom},
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nReport written to {RESULTS_DIR / f'{name}.md'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--self-test", action="store_true",
                   help="exercise the whole harness on CPU with the dry-run "
                        "backend; measures nothing, finds harness bugs for free")
    p.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct",
                   help="must FIT on one GPU: without a TP=1 run there is no "
                        "baseline and no speedup can be computed")
    p.add_argument("--quantization", default=None,
                   help="omit for an FP16 checkpoint; forcing a method on an "
                        "unquantized model fails at load")
    p.add_argument("--tensor-parallel", type=int, nargs="+", default=[1, 2])
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 64],
                   help="must include 1: that is where TP is expected to LOSE, "
                        "and a run that wins everywhere is suspect")
    p.add_argument("--requests", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup-requests", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--load-timeout", type=float, default=1800.0,
                   help="a 32B checkpoint can take many minutes to download and load")
    p.add_argument("--settle-seconds", type=float, default=20.0,
                   help="wait for the previous engine to release GPU memory")
    p.add_argument("--request-timeout", type=float, default=600.0)
    p.add_argument("--allow-slow-interconnect", action="store_true")
    args = p.parse_args()

    args.concurrency = sorted(set(args.concurrency))
    args.tensor_parallel = sorted(set(args.tensor_parallel))
    if args.self_test:
        args.requests = min(args.requests, 32)
        args.repeats = min(args.repeats, 2)
        args.warmup_requests = min(args.warmup_requests, 4)
        args.load_timeout = min(args.load_timeout, 90.0)
        args.settle_seconds = min(args.settle_seconds, 2.0)

    try:
        return asyncio.run(main_async(args))
    except Failure as exc:
        print(f"\nRefusing to report: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
