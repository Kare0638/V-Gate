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
Tensor-parallel configuration, and the harness helpers that decide whether a
measurement is worth reporting.

The GPU work itself cannot be tested here. What can be tested is everything
that would waste rented GPU time if it were wrong: that the setting reaches
vLLM, that an FP16 checkpoint can be configured at all, and that the guards
which refuse a bad measurement actually fire.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.bench_tensor_parallel import interconnect_of, read_kv_cache_blocks
from vgate.config import ModelConfig


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def test_defaults_to_a_single_device():
    assert ModelConfig().tensor_parallel_size == 1


def test_zero_or_negative_is_rejected():
    for bad in (0, -1):
        with pytest.raises(ValueError, match="must be >= 1"):
            ModelConfig(tensor_parallel_size=bad)


def test_quantization_may_be_none():
    """
    Required for any FP16 checkpoint. The default is "awq", and forcing that on
    an unquantized model fails at load rather than falling back -- so a field
    that could not be cleared would make every FP16 model unloadable.
    """
    assert ModelConfig(quantization=None).quantization is None


def test_tensor_parallel_size_reaches_the_engine_arguments():
    """
    The plumbing, which is the part that would silently do nothing. A config
    field that never reaches AsyncEngineArgs leaves both runs on one GPU, and
    the report would then compare a configuration against itself and call the
    1.0x result a finding.
    """
    import asyncio

    import vgate.backends.vllm_backend as backend_module

    # load_model() grabs the running loop so generate() can hand work back to
    # it later. Depending on whatever loop an earlier test happened to leave
    # behind makes this pass alone and fail in a full run, which is how it
    # first showed up.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    captured = {}

    class FakeEngineArgs:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeAsyncLLMEngine:
        @staticmethod
        def from_engine_args(engine_args):
            return object()

    fake_vllm = type(sys)("vllm")
    fake_vllm.AsyncLLMEngine = FakeAsyncLLMEngine
    fake_arg_utils = type(sys)("vllm.engine.arg_utils")
    fake_arg_utils.AsyncEngineArgs = FakeEngineArgs
    fake_engine = type(sys)("vllm.engine")

    saved = {k: sys.modules.get(k) for k in
             ("vllm", "vllm.engine", "vllm.engine.arg_utils")}
    sys.modules["vllm"] = fake_vllm
    sys.modules["vllm.engine"] = fake_engine
    sys.modules["vllm.engine.arg_utils"] = fake_arg_utils
    try:
        backend_module.VLLMBackend().load_model(
            ModelConfig(tensor_parallel_size=2, quantization=None)
        )
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        asyncio.set_event_loop(None)
        loop.close()

    assert captured.get("tensor_parallel_size") == 2, (
        "tensor_parallel_size never reached AsyncEngineArgs; both runs would "
        "have used one GPU and the comparison would be meaningless"
    )


# --------------------------------------------------------------------------
# Harness guards
# --------------------------------------------------------------------------

TOPO = """\t GPU0\t GPU1\tCPU Affinity
GPU0\t X \t{link}\t0-31
GPU1\t{link}\t X \t0-31
"""


@pytest.mark.parametrize("link", ["NV12", "PIX", "PHB", "SYS"])
def test_interconnect_is_read_from_the_topology_matrix(link):
    """
    The single most important field in the report. Tensor parallelism ends
    every layer with an all-reduce, so NVLink versus PCIe is roughly a tenfold
    difference in what the number means -- and a result quoted without it
    cannot be compared to a rerun anywhere else.
    """
    assert interconnect_of(TOPO.format(link=link)) == link


def test_missing_topology_is_reported_as_unknown_not_guessed():
    assert interconnect_of(None) is None
    assert interconnect_of("nvidia-smi: command not found") is None


@pytest.mark.parametrize("line, expected", [
    ("INFO 06-01 12:00:00 # GPU blocks: 12,345, # CPU blocks: 4096", 12345),
    ("GPU KV cache size: 78,900 tokens", 78900),
    ("no such line here", None),
])
def test_kv_cache_blocks_are_read_from_the_server_log(tmp_path, line, expected):
    """
    The mechanism behind the result. Sharding frees device memory, that memory
    becomes KV cache, and KV cache bounds concurrency -- so this ratio is what
    explains a throughput difference rather than leaving it to be guessed at.
    """
    log = tmp_path / "server.log"
    log.write_text(line, encoding="utf-8")
    assert read_kv_cache_blocks(log) == expected


def test_missing_log_does_not_raise():
    assert read_kv_cache_blocks(Path("/nonexistent/server.log")) is None
