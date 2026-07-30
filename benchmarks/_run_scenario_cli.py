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
Run exactly one run_report.py scenario and write its JSON result to disk.

Exists so a real (GPU) benchmark run — where each scenario pays a multi-minute
model load — can be driven as a sequence of short-lived tool invocations
instead of one long-running process.

Usage:
    python benchmarks/_run_scenario_cli.py \\
        --title "Baseline (max_batch_size=8, all-unique prompts)" \\
        --batch-size 8 --prompts unique --engine-type vllm \\
        --concurrency 8 --requests 40 --out benchmarks/results/vllm/baseline.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.run_report import (  # noqa: E402
    REPEATED_PROMPTS,
    UNIQUE_PROMPTS,
    _run_scenario,
)


async def _main(args: argparse.Namespace) -> None:
    prompts = REPEATED_PROMPTS if args.prompts == "repeated" else UNIQUE_PROMPTS

    if args.engine_type == "dry-run":
        env = {"VGATE_DRY_RUN": "true"}
        startup_timeout_s = 30.0
    else:
        env = {
            "VGATE_MODEL__ENGINE_TYPE": args.engine_type,
            "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
        }
        startup_timeout_s = 300.0

    env["VGATE_BATCH__MAX_BATCH_SIZE"] = str(args.batch_size)
    env["VGATE_BATCH__MAX_WAIT_TIME_MS"] = "50"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = out_path.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    result = await _run_scenario(
        args.title, env, args.concurrency, args.requests, prompts,
        startup_timeout_s, log_dir,
    )
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run a single benchmark scenario")
    parser.add_argument("--title", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--prompts", choices=["unique", "repeated"], required=True)
    parser.add_argument("--engine-type", default="vllm", choices=["dry-run", "vllm", "sglang"])
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
