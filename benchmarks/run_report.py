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
Generates benchmarks/results/baseline.md by spawning a real V-Gate server
process per scenario (fresh process per scenario avoids Prometheus's global
registry rejecting a second app instance in the same process) and driving
load against it with bench_load.run_load_test().

Scenarios:
    1. Baseline: default batch config, all-unique prompts (no cache/dedup help).
    2. Batch size sweep: same workload, max_batch_size in {1, 4, 16, 32}.
    3. Cache impact: a small pool of repeated prompts vs. the all-unique baseline.

In dry-run mode (default, no GPU) the backend has ~0 real compute cost, so
batch size would show no effect on its own. VGATE_DRYRUN_SIMULATED_LATENCY_MS
gives the dry-run backend a synthetic per-call cost (documented, clearly
labeled in the report) so the batching scenarios are actually comparable.
This synthetic cost is never applied to vLLM/SGLang backends and never
affects the test suite (the knob defaults to 0 and only this script sets it).

Usage:
    python benchmarks/run_report.py
    python benchmarks/run_report.py --engine-type vllm   # requires GPU + model
"""

import argparse
import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bench_load import format_markdown, run_load_test  # noqa: E402

RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
PORT = 8091
BASE_URL = f"http://127.0.0.1:{PORT}"

UNIQUE_PROMPTS = [
    f"Explain distributed systems concept #{i} in one sentence." for i in range(64)
]
REPEATED_PROMPTS = [
    "Explain the concept of machine learning in one paragraph.",
    "Write a Python function that computes the Fibonacci sequence.",
    "What are the benefits of using a load balancer?",
]


def _spawn_server(env_overrides: Dict[str, str], log_path: Optional[Path] = None) -> subprocess.Popen:
    env = os.environ.copy()
    env.update({
        "VGATE_SERVER__PORT": str(PORT),
        "VGATE_SECURITY__ENABLED": "false",
        "VGATE_TRACING__ENABLED": "false",
        "VGATE_LOGGING__LEVEL": "WARNING",
    })
    env.update(env_overrides)
    log_file = open(log_path, "w", encoding="utf-8") if log_path else subprocess.DEVNULL
    return subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        # New session/process group: vLLM spawns its own EngineCore worker
        # process via multiprocessing, which does not reliably die when only
        # the main.py PID is signaled. Killing the whole group on teardown
        # avoids leaking a GPU-memory-holding zombie between scenarios.
        start_new_session=True,
    )


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """Kill the server's whole process group, not just the main.py PID, so
    vLLM's spawned EngineCore worker (a separate process) can't outlive it
    and leak GPU memory into the next scenario."""
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


async def _wait_healthy(timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            try:
                async with session.get(
                    f"{BASE_URL}/health", timeout=aiohttp.ClientTimeout(total=1)
                ) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                pass
            await asyncio.sleep(0.3)
    raise RuntimeError("Server did not become healthy in time")


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


async def _run_scenario(
    title: str,
    env_overrides: Dict[str, str],
    concurrency: int,
    total_requests: int,
    prompts: List[str],
    startup_timeout_s: float = 20.0,
    log_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    log_path = (log_dir / f"{_slug(title)}.log") if log_dir else None
    proc = _spawn_server(env_overrides, log_path)
    try:
        await _wait_healthy(startup_timeout_s)
        result = await run_load_test(
            base_url=BASE_URL,
            concurrency=concurrency,
            total_requests=total_requests,
            prompts=prompts,
            max_tokens=64,
        )
        result["title"] = title
        result["env"] = {k: v for k, v in env_overrides.items() if not k.startswith("VGATE_DRYRUN")}
        return result
    finally:
        _terminate_process_group(proc)


async def main_async(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.engine_type == "dry-run":
        base_env = {
            "VGATE_DRY_RUN": "true",
            "VGATE_DRYRUN_SIMULATED_LATENCY_MS": str(args.simulated_latency_ms),
        }
        startup_timeout_s = 30.0
    else:
        base_env = {
            "VGATE_MODEL__ENGINE_TYPE": args.engine_type,
            # vLLM defaults pinned memory to off under WSL2 out of caution; this
            # environment's kernel (>= 4.19.121) and torch.cuda pin_memory() both
            # verified working, so it's safe to opt back in here. No-op elsewhere.
            "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
        }
        # Real engines pay full model load + KV cache warmup per scenario
        # (each scenario is a fresh subprocess); this took ~3 minutes for a
        # 1.5B AWQ model on an RTX 3060 laptop GPU.
        startup_timeout_s = 300.0

    scenarios: List[Dict[str, Any]] = []

    scenarios.append(await _run_scenario(
        "Baseline (max_batch_size=8, all-unique prompts)",
        {**base_env, "VGATE_BATCH__MAX_BATCH_SIZE": "8", "VGATE_BATCH__MAX_WAIT_TIME_MS": "50"},
        args.concurrency, args.requests, UNIQUE_PROMPTS, startup_timeout_s, log_dir,
    ))

    for batch_size in (1, 4, 16, 32):
        scenarios.append(await _run_scenario(
            f"Batch size sweep: max_batch_size={batch_size}",
            {**base_env, "VGATE_BATCH__MAX_BATCH_SIZE": str(batch_size), "VGATE_BATCH__MAX_WAIT_TIME_MS": "50"},
            args.concurrency, args.requests, UNIQUE_PROMPTS, startup_timeout_s, log_dir,
        ))

    scenarios.append(await _run_scenario(
        "Cache impact: 3 repeated prompts (high reuse) vs. baseline's all-unique prompts",
        {**base_env, "VGATE_BATCH__MAX_BATCH_SIZE": "8", "VGATE_BATCH__MAX_WAIT_TIME_MS": "50"},
        args.concurrency, args.requests, REPEATED_PROMPTS, startup_timeout_s, log_dir,
    ))

    lines = [
        "# V-Gate Benchmark Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Engine: {args.engine_type}",
        f"Concurrency: {args.concurrency}, requests per scenario: {args.requests}",
        "",
    ]
    if args.engine_type == "dry-run":
        lines += [
            "> **Dry-run baseline.** No GPU/model is used; the backend is a mock that echoes "
            f"a fixed response after a synthetic delay of `{args.simulated_latency_ms}ms + 2ms * max_tokens` "
            "per batch call (see `VGATE_DRYRUN_SIMULATED_LATENCY_MS` in "
            "`vgate/backends/base.py`). This isolates the batcher/cache/HTTP-layer behavior "
            "from real inference cost and is **not** a GPU throughput measurement. A "
            "vLLM/SGLang single-worker baseline on real hardware is still needed "
            "(`--engine-type vllm|sglang`) before quoting real tokens/sec numbers.",
            "",
        ]
    for s in scenarios:
        lines.append(format_markdown(s, title=s["title"]))

    md_path = RESULTS_DIR / "baseline.md"
    json_path = RESULTS_DIR / "baseline.json"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(scenarios, indent=2), encoding="utf-8")
    print(f"Report written to {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate the V-Gate benchmark report")
    parser.add_argument("--engine-type", default="dry-run", choices=["dry-run", "vllm", "sglang"])
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument(
        "--simulated-latency-ms", type=int, default=15,
        help="Dry-run only: synthetic per-batch-call base latency in ms",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
