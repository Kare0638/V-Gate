# =============================================================================
# V-Gate Dockerfile — GPU & CPU targets
# =============================================================================
# GPU:  docker build --target vgate-gpu -t vgate:latest .
# CPU:  docker build --target vgate-cpu -t vgate:cpu .
# =============================================================================

# -----------------------------------------------------------------------------
# GPU target: production inference with vLLM
# Base image includes CUDA, PyTorch, vLLM, FastAPI
# -----------------------------------------------------------------------------
FROM vllm/vllm-openai:latest AS vgate-gpu

# pydantic-settings is not bundled in the base image
RUN pip install --no-cache-dir pydantic-settings>=2.0.0

WORKDIR /app

COPY requirements.txt ./
COPY config.yaml ./
COPY vgate/ ./vgate/
COPY main.py ./

ENV VGATE_CONFIG_PATH=/app/config.yaml

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=300s \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["python3", "-m", "uvicorn"]
CMD ["main:app", "--host", "0.0.0.0", "--port", "8000"]

# -----------------------------------------------------------------------------
# CPU target: CI / testing / dry-run mode (no GPU required)
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS vgate-cpu

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml ./
COPY vgate/ ./vgate/
COPY main.py ./

ENV VGATE_DRY_RUN=true
ENV VGATE_CONFIG_PATH=/app/config.yaml

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=3 --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["python", "-m", "uvicorn"]
CMD ["main:app", "--host", "0.0.0.0", "--port", "8000"]
