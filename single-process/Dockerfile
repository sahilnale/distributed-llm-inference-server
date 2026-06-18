# Base image: NVIDIA's official PyTorch container.
# This gives us CUDA 12.1 + cuDNN + PyTorch pre-installed and tested together.
# Much more reliable than installing CUDA yourself on Ubuntu.
FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

# Install system dependencies
# --no-install-recommends keeps the image smaller
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (before src/).
# Docker layer caches this step — if requirements.txt hasn't changed,
# rebuilds skip straight to copying src/. Saves minutes on rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY benchmarks/ ./benchmarks/

# Prometheus multiprocess metrics directory
# Needs to exist before the server starts
RUN mkdir -p /tmp/prometheus_multiproc

ENV PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc

# Default environment — all overridable via docker-compose or -e flags
ENV MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct
ENV NUM_GPUS=1
ENV BATCH_WINDOW_MS=50
ENV MAX_BATCH_SIZE=8
ENV REDIS_URL=redis://redis:6379

EXPOSE 8000

# Run uvicorn from src/ so relative imports work
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
