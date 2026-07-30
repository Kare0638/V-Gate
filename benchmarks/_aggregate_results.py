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
Combine per-scenario JSON files written by _run_scenario_cli.py into a single
markdown report. Companion to _run_scenario_cli.py's split-into-many-processes
workflow (used when scenarios are too slow to fit in one process, e.g. a real
GPU backend where each scenario pays a multi-minute model load).

Usage:
    python benchmarks/_aggregate_results.py \\
        --title "V-Gate vLLM Benchmark (RTX 3060 Laptop, real GPU)" \\
        --notes "Some caveat text..." \\
        --out benchmarks/results/vllm_baseline.md \\
        benchmarks/results/vllm/batch_1.json benchmarks/results/vllm/batch_8.json ...
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bench_load import format_markdown  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Aggregate scenario JSON files into a markdown report")
    parser.add_argument("json_files", nargs="+", help="Scenario JSON files, in report order")
    parser.add_argument("--title", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    lines = [
        f"# {args.title}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    if args.notes:
        lines += [args.notes, ""]

    for path in args.json_files:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        lines.append(format_markdown(result, title=result.get("title", Path(path).stem)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
