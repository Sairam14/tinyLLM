"""
FastAPI application for serving GermanLM.

Endpoints:
- GET /health — Health check
- POST /v1/tokenize — Tokenize text
- POST /v1/generate — Generate text (with streaming support)
- GET /metrics — Prometheus metrics
"""

from contextlib import asynccontextmanager
import os
import time
from typing import Optional, AsyncIterator
import json

import torch
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from prometheus_client import generate_latest, REGISTRY

from tinyllm.config import ModelConfig
from tinyllm.model import GermanLM
from tinyllm.tokenizer import GermanTokenizer
from tinyllm.generate import KVCacheGenerator
from tinyllm.serving.schemas import (
    TokenizeRequest,
    TokenizeResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)
from tinyllm.serving.middleware import (
    MetricsMiddleware,
    get_api_key,
    check_rate_limit,
    model_loaded,
    generation_tokens,
    generation_latency,
)


# Global state
class AppState:
    model: Optional[GermanLM] = None
    tokenizer: Optional[GermanTokenizer] = None
    generator: Optional[KVCacheGenerator] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, unload on shutdown."""
    # Startup
    print("Loading model and tokenizer...")

    # Load model from checkpoint (replace with your checkpoint path)
    checkpoint_path = os.environ.get("MODEL_CHECKPOINT", "checkpoints/step_0000000_final.pt")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "tokenizer_32k.json")

    try:
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model_config = ModelConfig.from_dict(checkpoint["model_config"])
            app_state.model = GermanLM(model_config)
            app_state.model.load_state_dict(checkpoint["model"])
            app_state.model = app_state.model.to(app_state.device)
            app_state.model.eval()
            print(f"✓ Loaded model from {checkpoint_path}")
        else:
            # Create dummy model for testing
            model_config = ModelConfig(
                vocab_size=1000,
                d_model=256,
                n_heads=4,
                n_layers=2,
                max_seq_len=512,
            )
            app_state.model = GermanLM(model_config)
            app_state.model = app_state.model.to(app_state.device)
            print(f"⚠ Using dummy model (checkpoint not found at {checkpoint_path})")

        if os.path.exists(tokenizer_path):
            app_state.tokenizer = GermanTokenizer()
            app_state.tokenizer.load(tokenizer_path)
            print(f"✓ Loaded tokenizer from {tokenizer_path}")
        else:
            # Train dummy tokenizer
            app_state.tokenizer = GermanTokenizer(vocab_size=1000)
            corpus = [
                "Das ist ein Test.",
                "Guten Morgen!",
                "Dies ist ein längerer Text für das Tokenizer-Training.",
            ]
            app_state.tokenizer.train(corpus, verbose=False)
            print(f"⚠ Using dummy tokenizer")

        app_state.generator = KVCacheGenerator(
            app_state.model,
            app_state.tokenizer,
            device=app_state.device,
            max_new_tokens=100,
        )
        model_loaded.set(1)
        print(f"✓ Ready on {app_state.device}")

    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        model_loaded.set(0)
        raise

    yield

    # Shutdown
    model_loaded.set(0)
    print("Shutting down...")


# Create app with lifespan
app = FastAPI(
    title="tinyLLM",
    description="Production-grade German language model serving",
    version="0.1.0",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(MetricsMiddleware)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=app_state.model is not None,
        device=app_state.device,
    )


@app.post("/v1/tokenize", response_model=TokenizeResponse, tags=["tokenize"])
async def tokenize(
    request: TokenizeRequest,
    api_key: str = Depends(get_api_key),
) -> TokenizeResponse:
    """Tokenize text."""
    if app_state.tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded")

    tokens = app_state.tokenizer.encode(request.text, add_special_tokens=False)
    token_strings = [
        app_state.tokenizer.id_to_token(token_id) or f"<unk:{token_id}>"
        for token_id in tokens
    ]

    return TokenizeResponse(
        tokens=tokens,
        token_strings=token_strings,
        count=len(tokens),
    )


@app.post("/v1/generate", response_model=GenerateResponse, tags=["generate"])
async def generate(
    request: GenerateRequest,
    api_key: str = Depends(check_rate_limit),
) -> GenerateResponse:
    """Generate text (non-streaming)."""
    if app_state.generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        with torch.no_grad():
            generated_text = app_state.generator.generate(
                request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
            )

        # Count generated tokens
        generated_tokens_list = app_state.tokenizer.encode(
            generated_text, add_special_tokens=False
        )
        num_tokens_generated = len(generated_tokens_list) - len(
            app_state.tokenizer.encode(request.prompt, add_special_tokens=False)
        )
        num_tokens_generated = max(0, num_tokens_generated)  # Ensure non-negative

        latency = time.time() - start_time
        generation_tokens.labels(method="generate").inc(num_tokens_generated)
        generation_latency.observe(latency)

        return GenerateResponse(
            text=generated_text,
            tokens_generated=num_tokens_generated,
            generation_time_ms=latency * 1000,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/v1/generate/stream", tags=["generate"])
async def generate_stream(
    request: GenerateRequest,
    api_key: str = Depends(check_rate_limit),
) -> StreamingResponse:
    """Generate text with streaming (Server-Sent Events)."""
    if app_state.generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    async def event_generator() -> AsyncIterator[str]:
        """Generate SSE events."""
        start_time = time.time()

        try:
            with torch.no_grad():
                for token_str in app_state.generator.stream_generate(
                    request.prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                ):
                    # Send token as SSE
                    data = json.dumps({
                        "delta": token_str,
                        "finish_reason": None,
                    })
                    yield f"data: {data}\n\n"

                    generation_tokens.labels(method="generate_stream").inc()

            # Send finish event
            latency = time.time() - start_time
            finish_data = json.dumps({
                "delta": "",
                "finish_reason": "stop",
            })
            yield f"data: {finish_data}\n\n"

            generation_latency.observe(latency)

        except Exception as e:
            error_data = json.dumps({
                "error": str(e),
                "finish_reason": "error",
            })
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/metrics", tags=["monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest(REGISTRY)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
