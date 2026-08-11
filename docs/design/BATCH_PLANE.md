# V-Gate Multimedia Batch Compute Plane: Design And Task Breakdown

> **Status**: Design proposal. No part of the batch compute plane is implemented. Every module, endpoint, configuration key, state transition, and metric below describes intended work, not current runtime behavior.
>
> **Purpose**: turn the Priority 1 checklist in [README.md](../../README.md) into one ordered, testable implementation plan without creating a second scheduler alongside Daft.

---

## 1. Scope And Relationship To Other Documents

[README.md](../../README.md) holds the authoritative priority ordering across the online and batch planes. [ROADMAP.md](../../ROADMAP.md) covers the online serving plane. This document is the detailed working plan for the multimedia batch compute plane.

The first workload is deliberately narrow:

```text
video manifest
  -> URI and row validation
  -> video metadata, decode, and frame sampling
  -> batched multimodal captioning
  -> per-video results and dead-letter rows
  -> Parquet output plus job metadata
```

The batch plane is throughput-oriented and asynchronous. It does not run Daft, Ray, video decoding, or batch-model inference on the latency-sensitive chat/streaming request path.

The first release supports one server-defined pipeline, `video_caption`. It is not a general DAG engine and does not execute user-provided Python.

---

## 2. Architecture Decisions

These decisions replace the earlier idea of separate "Ray executor" and "Daft pipeline" scheduling layers.

### 2.1 Ray Jobs Owns Job Lifecycle; Daft Owns Pipeline Execution

There is one data-plane scheduler:

- **Ray Jobs** starts, observes, and stops a cluster-managed job entrypoint on an external Ray cluster.
- The job entrypoint configures the **Daft Ray Runner** and materializes the complete Daft query.
- Daft owns partition scheduling, data movement, decode/sampling operators, batching, GPU UDF placement, and Parquet writes for that query.
- A stateful `@daft.cls(gpus=1)` UDF loads the multimodal model once per UDF actor and reuses it across batches within the query.

V-Gate will not build a parallel layer of direct `@ray.remote` CPU tasks or custom GPU actors around the same Daft DataFrame. That would duplicate scheduling, obscure ownership of retries and cancellation, and introduce unnecessary transfers between Daft partitions and a second actor graph.

The local and distributed modes run the same pipeline definition:

```text
local mode       -> Daft native runner
distributed mode -> Ray Job entrypoint -> Daft Ray Runner
```

The model-reuse guarantee is intentionally scoped to one materialized Daft query. Cross-job model residency is not promised. If cold-start measurements later justify a separate long-lived inference service, that is a new design decision rather than an accidental property of this MVP.

### 2.2 The API Is A Control Surface, Not An Executor

The existing FastAPI service will mount a jobs router so it can reuse authentication, rate limiting, logging, and metrics middleware. The gateway may validate a request and update job metadata, but it must never import Ray or Daft and must never execute a media pipeline.

A separate V-Gate job worker owns submission and reconciliation:

- in local mode, it launches and supervises the Daft-native job entrypoint;
- in Ray mode, it submits that same entrypoint through the Ray Jobs API and reconciles Ray status into the V-Gate store;
- it observes cancellation requests and stops the local process or Ray Job;
- gateway restart does not terminate work already accepted by the worker or Ray cluster.

The worker is a control-plane process. Daft, through its selected runner, remains the data-plane executor.

### 2.3 SQLite Stores Metadata, Not Results

The first runtime store is SQLite in WAL mode behind a `JobStore` protocol. `InMemoryJobStore` exists only as a unit-test implementation; it is never a shipped runtime phase and must not be selectable in production configuration.

SQLite stores bounded job metadata such as:

- job ID, owner ID, pipeline type, and canonical request hash;
- idempotency key and its scope;
- input URI, server-approved output URI, and runner;
- V-Gate status and external Ray submission ID;
- timestamps, attempt number, lease/claim metadata, progress counters, and a bounded error summary.

SQLite does **not** store manifests, frames, captions, Parquet payloads, dead-letter rows, model outputs, or unbounded logs. Those belong under `output_uri` in the configured filesystem or object store.

SQLite is a single-host MVP choice. Its file locking is not a supported multi-pod coordination mechanism on a network filesystem. A networked store such as Postgres is deferred until the Kubernetes topology actually requires it.

### 2.4 State Machine And Cancellation

The external state machine is:

```text
queued -> submitted -> running -> succeeded
   |          |           |
   |          |           +-> failed
   |          +--------------> failed
   +--------------------------> failed

queued -----------------------> cancelled
submitted/running -> cancelling -> cancelled
                         +------> failed
```

Rules:

- `claim_next()` atomically claims a `queued` job and records a lease; it is never implemented as `get()` followed by `update()`.
- `submitted` means a local child process has been accepted or Ray Jobs returned an external submission ID. It does not mean user code is already running.
- Cancellation of a queued job is immediate. Cancellation of a submitted/running job first records `cancelling`, then the worker requests process/Ray termination and reconciles the terminal state.
- If the output commit completed before cancellation won the race, `succeeded` is retained; committed output is not relabelled as cancelled.
- If stop fails or times out, the job becomes `failed` with a bounded cancellation error instead of remaining in `cancelling` forever.
- Terminal states are immutable except for an explicit future administrative repair operation. There is no resume endpoint in the MVP.

### 2.5 Idempotency Contract

`POST /v1/jobs` accepts an optional `Idempotency-Key` header.

- Scope is `(owner_id, Idempotency-Key)` for the store's configured retention period.
- The store saves a canonical hash of the validated request body in the same transaction that creates the job.
- Reusing a key with the same canonical body returns the original job and does not submit another local process or Ray Job, regardless of the original job's current status.
- Reusing a key with a different body returns `409 Conflict` and exposes neither the original body nor another owner's metadata.
- Omitting the key creates a new job.
- The V-Gate job ID is also used as, or deterministically mapped to, the Ray Jobs submission ID so reconciliation after a worker restart cannot submit a duplicate unknowingly.

This provides idempotent submission, not exactly-once execution. Runtime retries remain at-least-once and outputs must therefore be safe to recompute.

### 2.6 URI, Ownership, And Quota Boundaries

Media URIs are untrusted input. The API and worker enforce:

- a configured scheme allowlist and per-scheme roots/buckets/prefixes;
- local `file://` access only beneath an explicit development data root;
- object-store access only to configured buckets/prefixes using worker-side credentials;
- either no `http(s)` input in the first release or a strict host allowlist with redirect and private-address protections;
- a server-selected output root/prefix rather than arbitrary client-selected write destinations;
- owner checks on submit, list, get, cancel, and result-location access;
- global and per-owner limits for active jobs, manifest bytes/rows, media size/duration, sampled frames, output tokens, and retained metadata;
- a fixed pipeline enum and bounded parameters, never arbitrary module names, commands, runtime environments, or code.

Authentication credentials, signed URLs, stack traces, and raw media contents are not persisted in SQLite or emitted as metric labels. Object-store credentials are available only to the job worker/Ray runtime that needs them.

### 2.7 Checkpoints And Atomic Output Publication

Execution is at-least-once. Each attempt writes immutable intermediate and result files beneath a job-specific staging prefix. Retries use stable item IDs, such as `(job_id, video_id)`, to skip or overwrite already committed work deterministically.

Publication follows one of two storage-specific protocols:

- local filesystem: write a complete temporary result set and publish with an atomic rename;
- object store: write immutable attempt files, then write the final manifest/commit marker last and update the job's `output_uri` only after that marker succeeds.

Partially written staging data is never returned as a successful result. Permanently bad rows are written to a dead-letter Parquet dataset with bounded, sanitized error fields. Checkpoint and output manifests live in object/file storage; SQLite stores only their URI and bounded progress metadata.

---

## 3. Hardware And Compatibility Constraints

Two hardware profiles are in scope. Every measured claim must state which profile produced it.

| Profile | Hardware | Role |
|---|---|---|
| Local development | RTX 3060 Laptop GPU, 6GB VRAM | Control-plane work, CPU pipeline development, fake-captioner CI, local Ray cluster for CPU-worker scaling |
| Rented benchmark | 2x A100, rented per session | Task 0 model smoke test, real-GPU closed loop, batch-size sweep, 1-vs-2 GPU measurement |

Consequences:

- native-runner development, control-plane work, and CI use a deterministic fake captioner and require no GPU on either profile;
- the local 6GB device cannot host the batch multimodal model alongside the online serving model, so it is not used for real-GPU benchmarks;
- on the rented profile the online serving plane is still not run concurrently with batch GPU work, so that every benchmark number has a single attributable workload;
- the selected multimodal model, processor, precision, frame count/resolution, model length, batch size, and vLLM memory settings are proven on the rented profile during Task 0; quantization is not assumed, because 80GB-class devices can run the candidate models in BF16;
- the rented profile has two GPUs, so at most two `gpus=1` captioning UDF replicas. A 1-vs-2 GPU measurement is in scope; extrapolating it to larger clusters is not;
- GPU capacity is rented by the hour, so every GPU script must first be exercised locally against the fake captioner. Exploratory work does not belong on rented hardware.

Ray, Daft, vLLM, Python, CUDA, PyTorch, and model versions are a compatibility unit. The batch environment must be pinned separately from the lightweight gateway dependencies. A moving `latest` dependency is not acceptable evidence for a reproducible benchmark.

Both profiles are Ampere (SM 8.6 local, SM 8.0 rented), so one pinned batch dependency set is expected to cover both. Task 0 confirms this rather than assuming it. FP8 paths are out of scope because Ampere does not support them.

The rented profile is ephemeral. A benchmark run is only valid evidence if the report records the exact image, pinned dependency set, model revision, and GPU count, because the instance that produced it will not exist afterwards.

---

## 4. Target Process And Module Layout

### 4.1 Process Layout

```text
Process A: FastAPI gateway
  - existing online serving plane
  - /v1/jobs router
  - validates identity, quota, URI policy, and request shape
  - reads/writes bounded metadata through JobStore
  - never imports daft or ray

Process B: V-Gate job worker
  - atomically claims queued jobs from SQLite
  - local mode: starts/reconciles a Daft-native job process
  - Ray mode: submits/reconciles/stops jobs through Ray Jobs
  - updates only metadata, status, progress summary, and output URI

Process C: batch job entrypoint
  - receives a validated immutable JobSpec
  - selects Daft native runner or Daft Ray Runner
  - constructs and materializes the complete video_caption DataFrame
  - writes staging, dead-letter, and committed Parquet output

External Ray cluster, distributed mode only
  - Ray Jobs hosts the job driver
  - Daft Ray Runner schedules the DataFrame across workers
  - @daft.cls(gpus=1) replicas run model inference on GPU workers
```

### 4.2 Proposed Module Tree

```text
vgate/jobs/
  __init__.py
  models.py                 # JobSpec, JobMetadata, JobStatus
  store.py                  # JobStore protocol + SQLiteJobStore
  router.py                 # FastAPI control-plane endpoints
  service.py                # ownership, idempotency, quota, state transitions
  worker.py                 # standalone claim/submit/reconcile loop
  runners/
    base.py                 # lifecycle runner protocol
    local_process.py        # Daft-native child-process lifecycle
    ray_jobs.py             # Ray Jobs submission/status/stop adapter
  pipelines/
    video_caption.py        # one runner-neutral Daft query definition
    schemas.py              # input/result/dead-letter schemas
  operators/
    fake_captioner.py       # deterministic CPU implementation for CI
    multimodal_captioner.py # @daft.cls(gpus=1) batch UDF
  entrypoint.py             # immutable JobSpec -> configure runner -> materialize
```

`InMemoryJobStore` belongs under tests or test-only helpers, not in the runtime configuration surface.

### 4.3 API Contract

The planned MVP endpoints are:

```text
POST /v1/jobs
GET  /v1/jobs
GET  /v1/jobs/{job_id}
POST /v1/jobs/{job_id}/cancel
GET  /v1/jobs/{job_id}/results
```

`GET /results` returns committed result metadata and an authorized result URI or redirect; it does not proxy a video dataset or load Parquet into gateway memory.

Example submission:

```json
{
  "type": "video_caption",
  "input_uri": "s3://approved-input/videos/manifest.parquet",
  "frame_sample_interval_seconds": 1.0,
  "max_frames_per_video": 16,
  "max_output_tokens": 128
}
```

The server derives `owner_id`, `job_id`, and `output_uri`. A client cannot supply them to escape its namespace.

---

## 5. Ordered Task Breakdown

### Task 0: Version-Compatibility And Runnable Spike

**Goal**: prove the chosen APIs and hardware assumptions before production structure is added.

Steps 1-5 run on the local profile and need no GPU. Steps 6-8 run on the rented profile and should be batched into a single short session, with the scripts already exercised locally against the fake captioner.

The spike must:

1. Pin a compatibility matrix for Python, Ray, Daft, vLLM, PyTorch, CUDA, the model, and any video codec dependency.
2. Run one tiny Daft video query with the native runner and the same query with the Daft Ray Runner on a local/external test Ray cluster.
3. Verify the current stable APIs for video metadata/frame sampling, `daft.set_runner_ray`, `@daft.cls(gpus=1)`, batched class methods, Parquet output, and Ray Jobs submit/status/stop.
4. Verify that configuring the Ray runner never silently starts a local Ray cluster in distributed mode.
5. Run a fake stateful class UDF and prove initialization occurs once per UDF actor rather than once per row/batch.
6. Select and smoke-test one vLLM-supported multimodal model on the rented profile in BF16, with the online plane not running.
7. Record model load time, peak VRAM, accepted frame representation, maximum safe sampled frames/resolution, output-token limit, and the largest batch size that remains stable on one GPU.
8. Decide whether the batch adapter uses vLLM's offline or async engine inside the Daft actor; it must not reuse the online gateway's event-loop-bound engine instance.

**Deliverables**:

- a checked-in version/compatibility note and pinned batch dependency file or lock;
- a disposable spike script and captured commands/results;
- this document updated if any assumed API or resource model is wrong.

**Acceptance**: both runner modes complete a tiny fake query, Ray Jobs lifecycle calls work, and one narrow real-GPU caption succeeds on the rented profile within the recorded VRAM and batch-size envelope. No production endpoint is added in this task.

### Task 1: Daft-Native Vertical Slice

**Goal**: prove the complete data contract locally before adding control-plane code.

Implement one runner-neutral `video_caption` pipeline and a CLI/entrypoint using the Daft native runner:

```text
manifest -> validate -> metadata/decode/sample -> FakeCaptioner
         -> aggregate -> result/dead-letter Parquet -> commit marker
```

Input manifest fields:

| Field | Requirement |
|---|---|
| `video_id` | non-empty, unique within a job after canonicalization |
| `video_uri` | allowed scheme/root and syntactically valid |

Result rows include `video_id`, `caption`, sampled-frame count, duration, status, and a bounded error code/message. Permanently invalid rows go to the dead-letter dataset without aborting valid rows.

Fixtures include normal short videos, a corrupt video, a missing URI, a disallowed URI, and duplicate IDs. `FakeCaptioner` is deterministic and GPU-free.

**Acceptance**:

- one command produces committed result and dead-letter Parquet with fixed schemas;
- normal rows succeed while bad rows are isolated according to policy;
- no Daft/Ray dependency enters the gateway process;
- tests run without a GPU and produce deterministic output;
- rerunning the same job/attempt does not expose partial or duplicate committed rows.

### Task 2: SQLite Job Store, Independent Worker, And Job API

**Goal**: add a persistent control plane around the native vertical slice.

Implement:

- `JobStore` plus `SQLiteJobStore` in WAL mode, with a busy timeout and blocking calls kept off the FastAPI event loop;
- a stable owner principal derived from the authenticated key configuration and exposed to the jobs service without storing or propagating the raw API-key secret; when authentication is disabled, only one explicitly configured local-development owner is allowed;
- transactional job creation/idempotency and atomic `claim_next()` with lease metadata;
- the state machine in section 2.4, including `cancelling`;
- the endpoints in section 4.3 with owner isolation, URI validation, and initial quotas;
- a standalone worker process that claims jobs and launches/reconciles the Daft-native entrypoint;
- configuration for database path, worker polling/lease intervals, allowed inputs, output root, and quotas.

`InMemoryJobStore` is used only for fast protocol tests. SQLite is used by integration tests and the runnable local mode.

**Acceptance**:

- job metadata survives gateway and worker restarts;
- two workers cannot claim the same job;
- same owner/key/body returns the same job, while same owner/key/different body returns 409;
- one owner cannot list, read, cancel, or obtain result locations for another owner's job;
- SQLite contains only bounded metadata and output/checkpoint URIs, never result datasets;
- queued cancellation is immediate and running local cancellation reaches a terminal state;
- the gateway process neither imports Daft/Ray nor executes a pipeline.

### Task 3: Ray Jobs Lifecycle And Daft Ray Runner

**Goal**: run the Task 1 pipeline unchanged on an external Ray cluster.

Implement a Ray Jobs lifecycle adapter in the standalone worker:

- submit the batch entrypoint with a deterministic external submission ID;
- provide an immutable, validated JobSpec without embedding credentials in command-line arguments or logs;
- select a pinned Ray runtime environment/image;
- configure the entrypoint to connect to the intended cluster and use the Daft Ray Runner;
- map Ray pending/running/succeeded/stopped/failed states into the V-Gate state machine;
- stop the Ray Job when a V-Gate job is `cancelling`;
- reconcile existing submission IDs after worker restart before considering resubmission.

Daft executes the full query. This task does not add custom Ray CPU tasks, custom inference actors, or conversions to and from a separate Ray Dataset graph.

**Acceptance**:

- the same immutable JobSpec and pipeline definition produce equivalent schemas/results under native and Ray runners;
- gateway/worker restart does not duplicate an already submitted Ray Job;
- API cancellation calls Ray Jobs stop and reaches `cancelled` or a bounded `failed` outcome;
- stopping the FastAPI gateway does not stop an accepted Ray Job;
- a distributed run proves work is executed on the configured Ray workers, not an accidental local cluster.

### Task 4: Multimodal GPU Adapter

**Goal**: replace the fake captioner with real batched vLLM multimodal inference while preserving the Task 1 data contract.

Implement a batch-specific adapter rather than reusing the current text-only, event-loop-bound online `VLLMBackend`:

```python
@daft.cls(gpus=1, max_concurrency=1)
class MultimodalCaptioner:
    def __init__(self, model_config):
        # Load processor and one vLLM engine per Daft UDF actor.
        ...

    @daft.method.batch(batch_size=...)
    def caption(self, frames, prompt):
        ...
```

Exact type annotations and decorator arguments are pinned to the Daft version proven in Task 0.

Requirements:

- convert sampled frames into the selected model's supported vLLM multimodal prompt shape;
- use the model's processor/chat template rather than concatenating message strings manually;
- initialize model weights once per Daft class-UDF actor and reuse them across query batches;
- declare one GPU per replica explicitly and let replica count follow the GPUs the runtime actually exposes, so moving from one to two GPUs is a resource change rather than a code change;
- bound frames, resolution, model length, batch size, and output tokens before inference;
- use deterministic decoding for correctness comparisons unless a benchmark explicitly studies sampling;
- retain `FakeCaptioner` for CI and unit tests;
- never run the online serving model concurrently with batch GPU work on either hardware profile.

**Acceptance**:

- instrumentation proves one model initialization serves multiple batches in a query, asserted in CI against `FakeCaptioner` rather than only observed by hand on GPU hardware;
- fake and real adapters return the same output schema;
- one narrow real-GPU job succeeds on the rented profile within Task 0's recorded envelope;
- a batch-size sweep records throughput, peak VRAM, and latency at each step, so the batching claim rests on a curve rather than a single point;
- going from one to two GPUs requires configuration changes only; if any code change is needed, it is recorded as an architecture finding rather than quietly patched;
- the online text-serving tests remain unchanged and pass;
- no claim is made about cross-job model reuse, and GPU scaling claims stop at the measured 1-vs-2 result.

### Task 5: Correctness, Security, Reliability, And Metrics

**Goal**: make failure and trust boundaries observable and testable before benchmarking.

Correctness and reliability work:

- classify data errors separately from transient infrastructure errors;
- bound retries and backoff; never repeatedly retry a permanently corrupt video;
- use stable item IDs, checkpoints, staging prefixes, dead-letter output, and the atomic publication protocol in section 2.7;
- reconcile expired worker leases and existing Ray submissions after control-worker failure;
- define timeouts for submission, queueing, running, cancellation, and status reconciliation;
- inject worker exit, Ray job-driver exit, GPU UDF failure/OOM, unreadable media, and output-write failure;
- verify committed output is either complete or absent and that terminal status matches the commit outcome.

Security work:

- enforce scheme/root/bucket/host allowlists in both API validation and worker execution;
- re-resolve and revalidate paths/URIs at execution time rather than trusting only API-time checks;
- enforce owner access and global/per-owner quotas from section 2.6;
- sanitize error details and logs, and keep secrets out of JobSpec, SQLite, subprocess arguments, and Ray submission metadata;
- reject arbitrary pipeline names, runtime environments, commands, and code-bearing parameters.

Metrics use bounded labels only; `job_id`, `owner_id`, URI, model prompt, and error text are not labels. Planned series include:

```text
vgate_jobs_total{type,runner,status}
vgate_jobs_active{type,runner}
vgate_job_duration_seconds{type,runner,status}
vgate_job_rows_total{type,runner,status}
vgate_job_stage_duration_seconds{type,runner,stage}
vgate_job_retries_total{type,runner,stage,reason}
vgate_job_cancellations_total{runner,outcome}
vgate_job_cpu_seconds{type,runner}
vgate_job_gpu_seconds{type,runner}
```

Per-job details belong in structured logs/traces and the metadata API, not high-cardinality Prometheus labels.

**Acceptance**:

- failure-injection tests do not lose a job, leak another owner's data, or expose partial output as success;
- retries distinguish transient infrastructure failures from row-level data failures;
- cancellation cannot remain in `cancelling` indefinitely;
- a worker restart reconciles rather than duplicates running Ray work;
- URI and quota tests cover traversal, disallowed bucket/host, redirect/private-address behavior where applicable, and body-conflict idempotency;
- metrics and logs explain queue, decode, inference, retry, cancellation, and commit behavior without unbounded labels or secrets.

### Task 6: Reproducible Benchmarks And Evidence

**Goal**: produce evidence in the style of the existing checked-in benchmark reports.

Benchmark separately:

1. Daft native runner with `FakeCaptioner` as the single-machine baseline.
2. Daft Ray Runner with `FakeCaptioner`, first with one CPU worker and then N CPU workers.
3. Real-GPU `MultimodalCaptioner` runs on the rented 2x A100 profile: a batch-size sweep on one GPU, then a 1-vs-2 GPU comparison at the selected batch size.
4. Worker/Ray failure and recovery, including time to terminal reconciliation.

Items 1, 2, and 4 run on the local profile and need no rented hardware. Only item 3 does, and it should reuse the session already spent on Task 4 rather than booking another one.

Record dataset composition, warmup, repetitions, pinned versions, hardware, runner configuration, partition count, frame policy, model settings, and confidence/variance. Report wall time, videos/sec, frames/sec, per-stage p50/p95, queue time, retries, success rate, CPU/GPU time, model cold start, peak RAM, and peak VRAM.

The report must distinguish framework overhead from useful scaling. Ray-vs-native on one machine is not presented as a throughput win by default, and 1-vs-N CPU-worker results must not be described as multi-GPU scaling.

**Acceptance**:

- one documented command produces a Markdown/JSON report under `benchmarks/results/`;
- another developer can reproduce the fake-captioner comparison from pinned dependencies and fixtures;
- the report states which hardware profile produced each number, that online serving was not running, and which scaling claims were not measured;
- résumé numbers are copied only from this reproducible report.

---

## 6. Non-Goals For This Milestone

- A general DAG/workflow language or UI.
- User-provided Python, shell commands, Ray runtime environments, or arbitrary operators.
- A second direct-Ray task/actor scheduler around Daft.
- A resume endpoint or exactly-once execution guarantee.
- Cross-job GPU model residency or a standalone inference-worker service.
- Running online and batch GPU workloads concurrently on either hardware profile.
- Scaling claims beyond the measured 1-vs-2 GPU comparison, including extrapolation to larger clusters.
- Multi-node GPU clusters. The rented profile is one instance with two GPUs; distributed evidence comes from CPU workers on the local profile.
- A networked job database, KubeRay autoscaling, or multi-pod control plane.
- vLLM-Omni, model registry/canary rollout, custom C++/CUDA kernels, or training/fine-tuning.

These may become later milestones only after Task 6 identifies a concrete need or measured bottleneck.
