"""Middleware for API key auth, rate limiting, and Prometheus metrics."""

from typing import Optional, List
import os
import time
from functools import wraps

from fastapi import Depends, HTTPException, Request, Header
from prometheus_client import Counter, Histogram, Gauge

# Prometheus metrics
request_count = Counter(
    "llm_requests_total",
    "Total number of requests",
    ["endpoint", "status"],
)

generation_tokens = Counter(
    "llm_tokens_generated_total",
    "Total tokens generated",
    ["method"],
)

generation_latency = Histogram(
    "llm_generation_latency_seconds",
    "Generation latency in seconds",
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
)

model_loaded = Gauge(
    "llm_model_loaded",
    "Whether the model is loaded",
)

request_latency = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
)


def get_api_key(authorization: Optional[str] = Header(None)) -> str:
    """Extract and validate API key from Authorization header.

    Args:
        authorization: Authorization header value

    Returns:
        API key if valid

    Raises:
        HTTPException: If missing or invalid
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization format. Expected 'Bearer <key>'")

    api_key = authorization[7:]  # Strip "Bearer "

    # Load valid keys from environment
    valid_keys = os.environ.get("API_KEYS", "").split(",")
    valid_keys = [k.strip() for k in valid_keys if k.strip()]

    if api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return api_key


class MetricsMiddleware:
    """Track request metrics with Prometheus."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, request: Request, call_next):
        start_time = time.time()

        try:
            response = await call_next(request)
            status = response.status_code
        except Exception as e:
            status = 500
            raise

        latency = time.time() - start_time

        # Record metrics
        endpoint = request.url.path
        method = request.method
        request_count.labels(endpoint=endpoint, status=status).inc()
        request_latency.labels(method=method, endpoint=endpoint).observe(latency)

        return response


class RateLimiter:
    """Simple in-memory rate limiter (not production-grade, use Redis in production)."""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: dict = {}  # {key: [(timestamp, ...)])}

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed for key."""
        now = time.time()
        minute_ago = now - 60

        if key not in self.requests:
            self.requests[key] = []

        # Remove old requests
        self.requests[key] = [ts for ts in self.requests[key] if ts > minute_ago]

        if len(self.requests[key]) >= self.requests_per_minute:
            return False

        self.requests[key].append(now)
        return True


# Global rate limiter
rate_limiter = RateLimiter(requests_per_minute=60)


def check_rate_limit(api_key: str = Depends(get_api_key)) -> str:
    """Check rate limit for API key."""
    if not rate_limiter.is_allowed(api_key):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded: 60 requests per minute",
        )
    return api_key
