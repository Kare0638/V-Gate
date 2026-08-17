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
The dry-run backend's synthetic cost and capacity knobs.

Both exist only so benchmarks have something to measure, and both are read
from the environment at import time. They are covered here because their
default -- off -- is what keeps them out of every other test and out of any
real deployment, and nothing else would notice if that default changed.
"""

import importlib
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import vgate.backends.base as base


def reload_with(monkeypatch, **env):
    """Re-import the module so the import-time env reads happen again."""
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(base)


@pytest.fixture(autouse=True)
def restore_module():
    """Leave the module as the rest of the suite expects to find it."""
    yield
    importlib.reload(base)


def elapsed_for(backend, module, concurrency):
    params = backend.create_sampling_params(0.7, 0.9, 0)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        started = time.perf_counter()
        list(pool.map(lambda i: backend.generate([f"p{i}"], params), range(concurrency)))
        return time.perf_counter() - started


def test_both_knobs_are_off_by_default():
    """
    The default is what keeps a synthetic delay out of real deployments and out
    of the rest of the suite. Nothing else would catch it changing.
    """
    module = importlib.reload(base)
    assert module._DRYRUN_LATENCY_MS == 0
    assert module._DRYRUN_MAX_CONCURRENCY == 0
    assert module._dryrun_capacity is None


def test_generate_is_free_when_the_latency_knob_is_unset():
    module = importlib.reload(base)
    assert elapsed_for(module.DryRunBackend(), module, 4) < 0.1


def test_capacity_bounds_concurrent_generations(monkeypatch):
    """
    The property the scaling benchmark rests on: a worker's throughput is
    capacity/latency, independent of how many threads the host happens to give
    its executor. Without it, one worker absorbs everything the load generator
    sends until the executor fills, and 1 worker measures the same as 4.
    """
    module = reload_with(
        monkeypatch,
        VGATE_DRYRUN_SIMULATED_LATENCY_MS="100",
        VGATE_DRYRUN_MAX_CONCURRENCY="2",
    )
    backend = module.DryRunBackend()

    # Six calls through a capacity of two: three sequential rounds of 100ms.
    duration = elapsed_for(backend, module, 6)
    assert 0.28 < duration < 0.45, f"expected ~0.3s for 3 rounds, got {duration:.2f}s"


def test_without_the_capacity_knob_calls_do_not_queue(monkeypatch):
    """The same six calls run in one round when nothing bounds them."""
    module = reload_with(
        monkeypatch,
        VGATE_DRYRUN_SIMULATED_LATENCY_MS="100",
        VGATE_DRYRUN_MAX_CONCURRENCY="0",
    )
    duration = elapsed_for(module.DryRunBackend(), module, 6)
    assert duration < 0.25, f"calls queued unexpectedly: {duration:.2f}s"


def test_capacity_alone_costs_nothing_without_latency(monkeypatch):
    """
    The capacity gate lives inside the latency simulation, so a deployment that
    sets only the capacity knob pays no delay -- and the knob cannot slow down
    anything that was not already opting into synthetic cost.
    """
    module = reload_with(
        monkeypatch,
        VGATE_DRYRUN_SIMULATED_LATENCY_MS="0",
        VGATE_DRYRUN_MAX_CONCURRENCY="1",
    )
    assert elapsed_for(module.DryRunBackend(), module, 8) < 0.1
