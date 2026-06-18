"""Request and response schemas for serving API."""

from pydantic import BaseModel, Field
from typing import Optional, List


class TokenizeRequest(BaseModel):
    """Request to tokenize text."""

    text: str = Field(..., description="Text to tokenize")


class TokenizeResponse(BaseModel):
    """Response with tokenized text."""

    tokens: List[int] = Field(..., description="Token IDs")
    token_strings: List[str] = Field(..., description="Token strings")
    count: int = Field(..., description="Number of tokens")


class GenerateRequest(BaseModel):
    """Request to generate text."""

    prompt: str = Field(..., description="Input prompt")
    max_new_tokens: int = Field(default=100, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="Sampling temperature (0=greedy)")
    top_k: Optional[int] = Field(default=None, ge=1, description="Top-k filtering")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling probability")
    stream: bool = Field(default=False, description="Stream response (SSE)")


class GenerateResponse(BaseModel):
    """Response with generated text."""

    text: str = Field(..., description="Generated text (prompt + completion)")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    generation_time_ms: float = Field(..., description="Generation time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Compute device (cuda/cpu)")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional details")
