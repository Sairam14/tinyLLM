"""
Multi-stage Docker build for tinyLLM serving.

Stages:
1. builder: Install dependencies
2. runtime: Run API server (slim, no dev/training deps)
"""

# Stage 1: Builder
FROM nvidia/cuda:12.1.0-runtime-cudnn8-ubuntu22.04 as builder

WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy package definition
COPY pyproject.toml pyproject.toml

# Install dependencies (no dev deps)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .

# Stage 2: Runtime
FROM nvidia/cuda:12.1.0-runtime-cudnn8-ubuntu22.04

WORKDIR /app

# Install minimal Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY tinyllm/ tinyllm/
COPY scripts/ scripts/

# Create non-root user
RUN useradd -m -u 1000 tinyllm && chown -R tinyllm:tinyllm /app
USER tinyllm

# Environment
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV API_KEYS=test-key-1,test-key-2

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Start server
CMD ["python3", "-m", "uvicorn", "tinyllm.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
