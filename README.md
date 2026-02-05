# V-Gate

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

**V-Gate** is a high-performance AI model serving gateway built on [vLLM](https://github.com/vllm-project/vllm). It provides a unified, OpenAI-compatible API interface with enterprise-grade features including dynamic request batching, result caching, observability, and security.

Optimized for memory-constrained environments (e.g., RTX 3060/4060).

---

## Features

| Feature | Description |
|---------|-------------|
| **OpenAI-Compatible API** | Drop-in replacement for OpenAI API (`/v1/chat/completions`, `/v1/embeddings`) |
| **Dynamic Request Batching** | Aggregate concurrent requests for improved GPU utilization |
| **Result Caching** | LRU cache with batch-level deduplication |
| **Prometheus Metrics** | Full observability with `/metrics` endpoint |
| **Structured Logging** | JSON-formatted logs for production debugging |
| **API Key Authentication** | Bearer token validation with per-key rate limits |
| **Rate Limiting** | Sliding window algorithm to prevent abuse |
| **Configuration as Code** | YAML configuration with environment variable overrides |
| **Docker Ready** | Multi-stage build with GPU and CPU targets |

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
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VGATE_CONFIG_PATH` | Path to config file | `./config.yaml` |
| `VGATE_DRY_RUN` | Enable dry-run mode (no GPU) | `false` |
| `VGATE_SERVER__PORT` | Server port | `8000` |
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
├── vgate/
│   ├── __init__.py
│   ├── engine.py           # vLLM inference engine
│   ├── batcher.py          # Request batching logic
│   ├── cache.py            # LRU result cache
│   ├── config.py           # Configuration management
│   ├── logging_config.py   # Structured logging
│   ├── metrics.py          # Prometheus metrics
│   └── security.py         # Authentication & rate limiting
├── monitoring/
│   └── prometheus.yml      # Prometheus configuration
└── tests/
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
  - [ ] 4.2 Python Client SDK
  - [ ] 4.3 Kubernetes Deployment

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feat/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feat/amazing-feature`)
5. Open a Pull Request
