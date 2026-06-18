.PHONY: help install install-dev clean lint format test train-tokenizer train serve export-onnx export-gguf benchmark

PYTHON := python3
PIP := pip3
VENV := .venv
PYTHON_VENV := $(VENV)/bin/python
PIP_VENV := $(VENV)/bin/pip

help:
	@echo "tinyLLM — Production-Grade German LLM"
	@echo ""
	@echo "Available targets:"
	@echo "  make install            Install production dependencies"
	@echo "  make install-dev        Install dev + production dependencies"
	@echo "  make clean              Remove __pycache__, .pyc, .venv"
	@echo "  make lint               Run ruff + mypy"
	@echo "  make format             Format code with black + ruff"
	@echo "  make test               Run pytest suite"
	@echo "  make train-tokenizer    Train HF BPE tokenizer on German data"
	@echo "  make train              Run training loop (DDP-ready)"
	@echo "  make serve              Start FastAPI dev server"
	@echo "  make export-onnx        Export model to ONNX + INT8 quantization"
	@echo "  make export-gguf        Export model to GGUF (llama.cpp)"
	@echo "  make benchmark          Run inference benchmarks across all formats"

# Virtual environment
venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP_VENV) install --upgrade pip setuptools wheel

# Dependencies
install: venv
	$(PIP_VENV) install -e .

install-dev: venv
	$(PIP_VENV) install -e ".[dev]"

# Cleanup
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete
	rm -rf .mypy_cache .pytest_cache .ruff_cache build dist *.egg-info

clean-venv: clean
	rm -rf $(VENV)

# Code quality
lint:
	$(PYTHON_VENV) -m ruff check tinyllm/ tests/ scripts/
	$(PYTHON_VENV) -m mypy tinyllm/

format:
	$(PYTHON_VENV) -m black tinyllm/ tests/ scripts/
	$(PYTHON_VENV) -m ruff check --fix tinyllm/ tests/ scripts/

# Testing
test:
	$(PYTHON_VENV) -m pytest tests/ -v

test-cpu:
	$(PYTHON_VENV) -m pytest tests/ -v -m "not gpu"

test-gpu:
	$(PYTHON_VENV) -m pytest tests/ -v -m "gpu"

# Training and inference
train-tokenizer:
	$(PYTHON_VENV) scripts/train_tokenizer_hf.py

train:
	$(PYTHON_VENV) tinyllm/train.py

serve:
	$(PYTHON_VENV) -m uvicorn tinyllm.serving.app:app --reload --host 0.0.0.0 --port 8000

# Export formats
export-onnx:
	$(PYTHON_VENV) tinyllm/export/onnx_export.py

export-gguf:
	$(PYTHON_VENV) tinyllm/export/gguf_export.py

# Benchmarking
benchmark:
	$(PYTHON_VENV) scripts/benchmark.py

# Docker
docker-build-api:
	docker build -f Dockerfile -t tinyllm:latest .

docker-compose-up:
	docker compose up -d

docker-compose-down:
	docker compose down

docker-compose-logs:
	docker compose logs -f api

# Development helpers
dev-setup: install-dev format
	@echo "✓ Development environment ready"

check: lint test
	@echo "✓ All checks passed"
