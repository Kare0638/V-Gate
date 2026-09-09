# Runbook: TP=1 vs TP=2 on rented GPUs

Every command here is billed. The point of writing them down beforehand is
that thinking, debugging, and reading documentation are not.

Run the self-test locally **before** renting anything:

```bash
python benchmarks/bench_tensor_parallel.py --self-test
```

It drives the whole harness against the dry-run backend on CPU — spawn, load,
sweep, assertions, report. It measures nothing, and that is the point: harness
bugs found on an A100 are billed by the hour.

The harness has also been run against a real GPU and a real vLLM engine at
TP=1 ([report](../../benchmarks/results/tensor_parallel_rtx3060_tp1.md)) on a
6GB laptop card. That pass found three bugs the CPU self-test could not:

- `VGATE_MODEL__QUANTIZATION=null` produced the four-character **string**, not
  a null — every FP16 checkpoint, which is what the A100 run uses, would have
  failed at load.
- `VLLM_WSL2_ENABLE_PIN_MEMORY` was missing, which crashes vLLM under WSL2.
- Measuring from the harness process, which imports torch and vLLM, reported
  **19 req/s where a clean subprocess reported 33** against the same server.
  Every figure would have been ~40% low, and not necessarily by the same
  amount at TP=1 and TP=2 — corrupting the comparison, not just the scale.

Run against whatever GPU you have before renting one.

---

## Choosing the pod

**2× A100 SXM 80GB.** SXM rather than PCIe because tensor parallelism
all-reduces after every layer, on the critical path of every token — NVLink is
roughly ten times PCIe bandwidth, so on PCIe the measurement largely describes
the interconnect.

**Storage: a *network* volume, ~150 GB.**

|  | Volume disk | Network volume |
|---|---|---|
| After pod termination | **deleted** | kept |
| Billed while stopped | $0.20/GB/mo | $0.07/GB/mo |

The first row is the one that matters. You will want to terminate the pod the
moment the run finishes, and a volume disk takes the 65 GB of weights with it —
so a follow-up run re-downloads them *while paying A100 rates*.

Storage is noise next to the GPUs (~$0.35/day against ~$80–100/day). The only
number worth optimising is GPU hours, and the way to lose money here is
forgetting to terminate.

## Choosing the model

It must **fit on one GPU**. A 70B model forces TP=2, which sounds like the
right test and is the wrong one: with no TP=1 run there is no baseline, no
speedup can be computed, and the result degrades to "it ran".

`Qwen/Qwen2.5-32B-Instruct` at FP16 is ~65 GB: it fits on one 80 GB card and
leaves it short of KV cache, so both configurations are measurable on identical
work and the difference has somewhere to come from.

---

## On the pod

### 1. Record the machine before touching anything

```bash
nvidia-smi topo -m
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
```

`GPU0`↔`GPU1` must read `NV#`. If it reads `SYS` or `PHB`, the run is measuring
the wiring — the harness refuses these by default rather than producing a
number whose caveat nobody reads.

### 2. Keep weights on the persistent volume

```bash
export HF_HOME=/workspace/hf
echo 'export HF_HOME=/workspace/hf' >> ~/.bashrc
```

Without this, 65 GB lands on the container disk and disappears with the pod.

### 3. Set up

```bash
cd /workspace
git clone https://github.com/Kare0638/V-Gate.git && cd V-Gate
pip install -r requirements.txt aiohttp
python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"
```

### 4. Prove the harness runs *here* before the real one

```bash
python benchmarks/bench_tensor_parallel.py --self-test
```

Two minutes on CPU. It will not touch a GPU, and it fails fast if anything
about this box breaks the plumbing.

### 5. The measurement

```bash
python benchmarks/bench_tensor_parallel.py \
  --model Qwen/Qwen2.5-32B-Instruct \
  --tensor-parallel 1 2 \
  --concurrency 1 4 16 64 \
  --requests 128 --repeats 3 \
  2>&1 | tee /workspace/tp-run.log
```

The first load downloads ~65 GB; budget 20–40 minutes before anything is
measured. Concurrency **must** include 1 — that is where TP=2 is expected to
*lose*, and a configuration that wins everywhere usually means the two runs
were not doing equal work.

### 6. Get the results off the pod before terminating

```bash
cat benchmarks/results/tensor_parallel.md
# then, from your laptop:
#   runpodctl send benchmarks/results/tensor_parallel.{md,json}
```

### 7. Terminate

Not stop — **terminate**. A stopped pod still bills the volume disk at the
higher stopped rate, and a network volume keeps the weights either way.

---

## What the result should look like

If TP=2 wins at every concurrency **including 1**, be suspicious before being
pleased. One request cannot use a second device for anything and now pays an
all-reduce per layer, so it should be slightly *slower*. Winning everywhere
usually means the two runs were not doing equal work.

The expected shape:

| Concurrency | Expectation |
|---|---|
| 1 | TP=2 slightly **slower** — communication with nothing to overlap it |
| 4–16 | roughly even, crossing over |
| 64 | TP=2 clearly ahead — more KV cache means more sequences resident |

The report prints GPU KV cache blocks for each configuration, which is the
mechanism: sharding frees device memory, that memory becomes KV cache, and KV
cache is what bounds concurrency once a model is loaded.

## If something goes wrong

| Symptom | Cause |
|---|---|
| OOM at TP=1 | Model too large for one card. Lower `--max-model-len`, or use a smaller model — do **not** skip TP=1, it is the baseline. |
| Harness refuses to start | `GPU0↔GPU1` is `SYS`/`PHB`. Get a different pod; `--allow-slow-interconnect` proceeds but the number then describes the wiring. |
| `Refusing to report: N cache hit(s)` | Working as designed. Throughput would have counted work no GPU did. |
| Second config fails to load | Previous engine still holds GPU memory. Raise `--settle-seconds`. |
