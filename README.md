# V-Gate

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

**V-Gate** is a high-performance AI model serving gateway with pluggable inference backends. It currently supports [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang), while exposing a unified OpenAI-compatible API with enterprise-grade features including dynamic request batching, result caching, observability, and security.

Optimized for memory-constrained environments (e.g., RTX 3060/4060).

---

## Features

| Feature | Description |
|---------|-------------|
| **OpenAI-Compatible API** | Drop-in replacement for OpenAI API (`/v1/chat/completions`, `/v1/embeddings`) |
| **Dynamic Request Batching** | Aggregate concurrent requests for improved GPU utilization |
| **Result Caching** | LRU cache with batch-level deduplication |
| **Multi-Backend Inference** | Switch backend with `model.engine_type` (`vllm` / `sglang`) |
| **Built-in Benchmarking** | Compare backends with CLI tool and `/v1/benchmark` API |
| **Prometheus Metrics** | Full observability with `/metrics` endpoint |
| **Structured Logging** | JSON-formatted logs for production debugging |
| **API Key Authentication** | Bearer token validation with per-key rate limits |
| **Rate Limiting** | Sliding window algorithm to prevent abuse |
| **Configuration as Code** | YAML configuration with environment variable overrides |
| **Docker Ready** | Multi-stage build with GPU and CPU targets |
| **Python Client SDK** | `pip install` ready client with sync/async support |

---

## Quick Start

### Option 1: Docker (Recommended)

**GPU Mode (Production)**
```bash
# Build and run with GPU support
docker compose up vgate

# Or build manually
docker build --target vgate-gpu -t vgate:latest .
docker run --gpus all -p 8000:8000 --ipc=host vgate:latest
```

**CPU Mode (CI/Testing)**
```bash
# Run in dry-run mode (no GPU required)
docker compose --profile cpu up vgate-cpu

# Or build manually
docker build --target vgate-cpu -t vgate:cpu .
docker run -p 8000:8000 vgate:cpu
```

**Full Monitoring Stack**
```bash
# Start V-Gate + Prometheus + Grafana
docker compose --profile monitoring up

# Access:
# - V-Gate API:  http://localhost:8000
# - Prometheus:  http://localhost:9090
# - Grafana:     http://localhost:3000 (admin/admin)
```

### Option 2: Local Development

```bash
# Clone repository
git clone https://github.com/Kare0638/V-Gate.git
cd V-Gate

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server
python main.py
```

### Option 3: Isolated SGLang Environment (Recommended)

Use a dedicated virtual environment for `sglang[all]` to avoid dependency conflicts with your existing `vllm` environment.

```bash
cd V-Gate

# Create isolated env for SGLang
uv venv .venv-sglang --python 3.12

# Install base project deps
uv pip install --python .venv-sglang/bin/python -r requirements.txt

# Install SGLang full stack
uv pip install --python .venv-sglang/bin/python "sglang[all]==0.5.9"

# Optional: install test tools in this env
uv pip install --python .venv-sglang/bin/python pytest pytest-asyncio
```

---

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
# {"status":"ok","version":"0.3.2"}
```

### Chat Completions
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-vgate-xxxxx" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 100
  }'
```

### Prometheus Metrics
```bash
curl http://localhost:8000/metrics
```

### Statistics
```bash
curl http://localhost:8000/stats
```

### Inline Benchmark API
```bash
curl -X POST http://localhost:8000/v1/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["Explain KV cache in one paragraph."],
    "max_tokens": 128,
    "rounds": 3
  }'
```

### Python Client SDK

```bash
pip install ./vgate-client
```

```python
from vgate_client import VGate

# Synchronous client
client = VGate(base_url="http://localhost:8000", api_key="sk-vgate-dev-example")

response = client.chat.create(
    model="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    max_tokens=100,
)
print(response.choices[0].message.content)

# Embeddings
embedding = client.embeddings.create(model="mock-embedding-model", input="Hello world")

# Health check
health = client.health()

client.close()
```

```python
from vgate_client import AsyncVGate

# Asynchronous client
async with AsyncVGate(base_url="http://localhost:8000", api_key="sk-...") as client:
    response = await client.chat.create(
        model="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
        messages=[{"role": "user", "content": "Hello!"}],
    )
```

---

## Benchmark

### Compare Backends (CLI)

```bash
# Compare vLLM and SGLang in dry-run mode
PYTHONPATH=. VGATE_DRY_RUN=true python benchmarks/bench_compare.py --backends vllm sglang

# Run vLLM only, custom rounds/tokens
PYTHONPATH=. python benchmarks/bench_compare.py --backends vllm --rounds 5 --max-tokens 256

# JSON output for automation
PYTHONPATH=. python benchmarks/bench_compare.py --backends vllm sglang --output json
```

### Benchmark Current Server Backend

`POST /v1/benchmark` runs benchmark through the full server pipeline (batcher + cache + backend) using the active `model.engine_type`.

---

## Configuration

V-Gate uses a layered configuration system with the following priority:

**Environment Variables > YAML Config > Defaults**

### Configuration File (`config.yaml`)

```yaml
version: "0.3.2"

server:
  host: "0.0.0.0"
  port: 8000

model:
  model_id: "Qwen/Qwen2.5-1.5B-Instruct-AWQ"
  quantization: "awq"
  gpu_memory_utilization: 0.7
  max_model_len: 2048
  trust_remote_code: true
  enforce_eager: true
  engine_type: "vllm"  # "vllm" or "sglang"

batch:
  max_batch_size: 8
  max_wait_time_ms: 50.0

cache:
  enabled: true
  maxsize: 1000

logging:
  level: "INFO"
  json_format: true

security:
  enabled: false
  api_keys:
    - key: "sk-vgate-prod-xxxxx"
      name: "production"
      rate_limit: 100
  rate_limiting:
    enabled: true
    default_limit: 60
    window_seconds: 60
  exempt_paths:
    - "/health"
    - "/metrics"

benchmark:
  warmup_rounds: 1
  test_rounds: 3
  max_tokens: 128
  prompts:
    - "Explain the concept of machine learning in one paragraph."
    - "Write a Python function that computes the Fibonacci sequence."
    - "What are the benefits of using a load balancer?"
```

### Backend Selection

```bash
# Default backend: vllm
VGATE_MODEL__ENGINE_TYPE=vllm python main.py

# Switch to SGLang backend
VGATE_MODEL__ENGINE_TYPE=sglang python main.py
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VGATE_CONFIG_PATH` | Path to config file | `./config.yaml` |
| `VGATE_DRY_RUN` | Enable dry-run mode (no GPU) | `false` |
| `VGATE_SERVER__PORT` | Server port | `8000` |
| `VGATE_MODEL__ENGINE_TYPE` | Inference backend (`vllm`/`sglang`) | `vllm` |
| `VGATE_MODEL__MODEL_ID` | Model identifier | `Qwen/Qwen2.5-1.5B-Instruct-AWQ` |
| `VGATE_BATCH__MAX_BATCH_SIZE` | Max batch size | `8` |
| `VGATE_CACHE__MAXSIZE` | Cache size limit | `1000` |
| `VGATE_LOGGING__LEVEL` | Log level | `INFO` |
| `VGATE_LOGGING__JSON_FORMAT` | JSON log format | `true` |
| `VGATE_SECURITY__ENABLED` | Enable security | `false` |

---

## Docker Images

| Image | Base | Size | Use Case |
|-------|------|------|----------|
| `vgate:latest` | `vllm/vllm-openai:latest` | ~9GB | Production GPU inference |
| `vgate:cpu` | `python:3.12-slim` | ~220MB | CI/CD, testing, dry-run |

### Build Commands

```bash
# GPU image
docker build --target vgate-gpu -t vgate:latest .

# CPU image
docker build --target vgate-cpu -t vgate:cpu .
```

---

## Monitoring

### Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `vgate_requests_total` | Counter | Total requests by endpoint, method, status |
| `vgate_request_latency_seconds` | Histogram | Request latency distribution |
| `vgate_batch_size` | Histogram | Batch size distribution |
| `vgate_batch_processing_seconds` | Histogram | Batch processing time |
| `vgate_ttft_seconds` | Histogram | Time to first token |
| `vgate_tpot_seconds` | Histogram | Time per output token |
| `vgate_tokens_generated_total` | Counter | Total tokens generated |
| `vgate_cache_hits_total` | Counter | Cache hits |
| `vgate_cache_misses_total` | Counter | Cache misses |

### Grafana Dashboard

After starting the monitoring stack, import the V-Gate dashboard in Grafana:
1. Navigate to http://localhost:3000
2. Login with `admin/admin`
3. Add Prometheus data source: `http://prometheus:9090`
4. Create dashboards using `vgate_*` metrics

---

## Security

### API Key Authentication

```bash
# Request with API key
curl -H "Authorization: Bearer sk-vgate-prod-xxxxx" \
     http://localhost:8000/v1/chat/completions \
     -d '{"model": "qwen", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Rate Limit Headers

| Header | Description |
|--------|-------------|
| `X-RateLimit-Limit` | Maximum requests allowed |
| `X-RateLimit-Remaining` | Remaining requests in window |
| `X-RateLimit-Reset` | Unix timestamp when window resets |
| `Retry-After` | Seconds to wait (on 429 response) |

---

## Project Structure

```
V-Gate/
├── main.py                 # FastAPI application entry point
├── config.yaml             # Default configuration
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Service orchestration
├── requirements.txt        # Python dependencies
├── benchmarks/
│   ├── benchmark.py         # Single-engine benchmark entry
│   └── bench_compare.py     # Multi-backend benchmark comparison CLI
├── vgate/
│   ├── __init__.py
│   ├── engine.py           # Backend factory + engine wrapper
│   ├── batcher.py          # Request batching logic
│   ├── backends/
│   │   ├── base.py         # Inference backend protocol + dry-run backend
│   │   ├── vllm_backend.py # vLLM backend adapter
│   │   └── sglang_backend.py # SGLang backend adapter
│   ├── cache.py            # LRU result cache
│   ├── config.py           # Configuration management
│   ├── logging_config.py   # Structured logging
│   ├── metrics.py          # Prometheus metrics
│   └── security.py         # Authentication & rate limiting
├── vgate-client/               # Python Client SDK
│   ├── pyproject.toml
│   ├── vgate_client/
│   │   ├── __init__.py
│   │   ├── client.py           # Sync & async clients
│   │   ├── models.py           # Request/response models
│   │   └── exceptions.py       # Error classes
│   └── tests/
├── monitoring/
│   └── prometheus.yml      # Prometheus configuration
└── tests/
    ├── test_backends.py
    ├── test_benchmark.py
    ├── test_batcher.py
    ├── test_cache.py
    ├── test_config.py
    ├── test_observability.py
    └── test_security.py
```

---

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
PYTHONPATH=. VGATE_DRY_RUN=true pytest tests/ -v

# Run specific test file
pytest tests/test_batcher.py -v

# Validate vLLM backend path
VGATE_DRY_RUN=true pytest tests/test_backends.py -k vllm -v

# Validate SGLang backend path (in .venv-sglang)
VGATE_DRY_RUN=true ./.venv-sglang/bin/pytest tests/test_backends.py -k sglang -v
```

### Code Style

```bash
# Format code
black .

# Lint
ruff check .
```

---

## Roadmap

- [x] **Phase 1**: Core MVP - Unified API Gateway
- [x] **Phase 2**: Performance Optimization
  - [x] 2.1 Dynamic Request Batching
  - [x] 2.2 Result Caching
  - [ ] 2.3 Multi-Worker Load Balancing
- [x] **Phase 3**: Production-Grade Features
  - [x] 3.1 Observability (Logging + Metrics)
  - [x] 3.2 Configuration as Code
  - [x] 3.3 Security & Access Control
- [x] **Phase 4**: Ecosystem & Deployment
  - [x] 4.1 Containerization (Docker)
  - [x] 4.2 Python Client SDK
  - [ ] 4.3 Kubernetes Deployment

---

## Compliance & Legal Disclaimer

1. **License**: This project is licensed under the Apache License 2.0.
2. **Model Terms**: V-Gate is an inference server. Users must separately adhere to the license terms of the underlying models (e.g., Qwen, LLaMA).
3. **Content Responsibility**: The author of V-Gate is NOT responsible for any content generated using this software. Users are fully responsible for the outputs and must ensure compliance with local safety laws and ethical guidelines.
4. **No Warranty**: This software is provided "as is", optimized for RTX 3060; use on other hardware is at your own risk.

See the [LICENSE](LICENSE) file for full license text.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feat/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feat/amazing-feature`)
5. Open a Pull Request
