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
One load-generating process, reporting its result as JSON on stdout.

Exists so bench_scaling.py can drive a gateway from several *processes* rather
than several coroutine groups. That distinction is the whole point of the
saturation experiment: task groups inside one interpreter share an event loop
and a GIL, so if they fail to add throughput it could equally mean the gateway
is saturated or that this one Python process is. Separate processes get their
own loop and can occupy other cores, so a flat total across them is evidence
about the server rather than about the client.

Not meant to be run by hand; use benchmarks/bench_load.py for that.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bench_load import run_load_test  # noqa: E402


async def main_async(args: argparse.Namespace) -> None:
    stamp = int(time.time() * 1000)
    prompts = [f"{args.tag} {stamp} {i}" for i in range(args.requests)]

    # Wait for a shared wall-clock start, so every client is loading the server
    # over the same window. Without it, interpreter startup alone staggers the
    # processes by hundreds of milliseconds, and a client that runs while the
    # others are still booting sees an uncontended server -- which inflates its
    # own rate and, once the parent sums per-process rates, inflates the total
    # for exactly the runs that have the most processes.
    if args.start_at:
        delay = args.start_at - time.time()
        if delay > 0:
            await asyncio.sleep(delay)

    # CLOCK_MONOTONIC, not time.time(), for the interval. On Linux this clock is
    # system-wide, so the parent can compare timestamps taken in different
    # processes -- and unlike the wall clock it cannot jump. It did jump: a
    # measured window came back as -53 seconds on this host, because WSL2
    # resynchronises its clock after the machine sleeps. The barrier above must
    # stay on the wall clock, since that is the only epoch the processes share
    # before they start, but nothing else here may.
    started_at = time.clock_gettime(time.CLOCK_MONOTONIC)
    result = await run_load_test(
        base_url=args.url,
        concurrency=args.concurrency,
        total_requests=args.requests,
        prompts=prompts,
        max_tokens=args.max_tokens,
    )
    ended_at = time.clock_gettime(time.CLOCK_MONOTONIC)

    json.dump(
        {
            # Kept for reference. The parent does NOT sum these: each is
            # requests over that process's own window, and summing windows that
            # do not coincide does not give throughput over any real interval.
            "requests_per_second": result["throughput"]["requests_per_second"],
            "requests": args.requests,
            "started_at": started_at,
            "ended_at": ended_at,
            "p95_s": result["latency"]["p95_s"],
            "failures": result["failures"],
            "cache_hits": result["cache"]["hits"],
            "deduplicated": result["batching"]["deduplicated"],
            "wall_time_s": result["wall_time_s"],
        },
        sys.stdout,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--requests", type=int, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--max-tokens", type=int, default=0)
    p.add_argument("--start-at", type=float, default=0.0,
                   help="unix timestamp to begin sending at, shared by all clients")
    asyncio.run(main_async(p.parse_args()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
