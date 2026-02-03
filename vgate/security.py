"""
V-Gate Security Module.

Provides API key authentication and rate limiting middleware for the V-Gate API gateway.

Features:
- API key validation via Bearer token
- Sliding window rate limiting per API key
- Configurable exempt paths (e.g., /health, /metrics)
- X-RateLimit-* response headers
"""
import time
from collections import defaultdict
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from vgate.config import SecurityConfig, APIKeyConfig
from vgate.logging_config import get_logger

logger = get_logger("vgate.security")


class RateLimiter:
    """
    Sliding window rate limiter.

    Tracks request timestamps per API key and enforces rate limits
    using a sliding window algorithm.
    """

    def __init__(self, window_seconds: int = 60):
        """
        Initialize the rate limiter.

        Args:
            window_seconds: Size of the sliding window in seconds.
        """
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _cleanup_old_requests(self, key: str, now: float) -> None:
        """Remove requests outside the current window."""
        cutoff = now - self.window_seconds
        self._requests[key] = [ts for ts in self._requests[key] if ts > cutoff]

    def is_allowed(self, key: str, limit: int) -> tuple[bool, dict]:
        """
        Check if a request is allowed under the rate limit.

        Args:
            key: The API key or identifier to check.
            limit: Maximum requests allowed in the window.

        Returns:
            Tuple of (allowed: bool, headers: dict) where headers contains
            X-RateLimit-* values.
        """
        now = time.time()
        self._cleanup_old_requests(key, now)

        current_count = len(self._requests[key])
        remaining = max(0, limit - current_count)
        reset_time = int(now + self.window_seconds)

        headers = {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_time),
        }

        if current_count >= limit:
            # Calculate retry-after based on oldest request in window
            if self._requests[key]:
                oldest = min(self._requests[key])
                retry_after = int(oldest + self.window_seconds - now) + 1
                headers["Retry-After"] = str(max(1, retry_after))
            else:
                headers["Retry-After"] = str(self.window_seconds)
            return False, headers

        # Record this request
        self._requests[key].append(now)
        headers["X-RateLimit-Remaining"] = str(remaining - 1)

        return True, headers

    def get_usage(self, key: str) -> dict:
        """Get current usage stats for a key."""
        now = time.time()
        self._cleanup_old_requests(key, now)
        return {
            "current_requests": len(self._requests[key]),
            "window_seconds": self.window_seconds,
        }


def extract_api_key(request: Request) -> Optional[str]:
    """
    Extract API key from the Authorization header.

    Expects format: Authorization: Bearer <api_key>

    Args:
        request: The FastAPI request object.

    Returns:
        The API key string if found, None otherwise.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return None

    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None

    return parts[1]


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware for API key authentication and rate limiting.

    This middleware:
    1. Checks if the request path is exempt from authentication
    2. Validates the API key from the Authorization header
    3. Enforces rate limits per API key
    4. Adds X-RateLimit-* headers to responses
    """

    def __init__(self, app, config: SecurityConfig):
        """
        Initialize the security middleware.

        Args:
            app: The FastAPI application.
            config: Security configuration from VGateConfig.
        """
        super().__init__(app)
        self.config = config
        self.key_map: dict[str, APIKeyConfig] = {k.key: k for k in config.api_keys}
        self.limiter = RateLimiter(config.rate_limiting.window_seconds)

        logger.info(
            "Security middleware initialized",
            extra={"extra_data": {
                "enabled": config.enabled,
                "api_keys_count": len(config.api_keys),
                "rate_limiting_enabled": config.rate_limiting.enabled,
                "exempt_paths": config.exempt_paths,
            }}
        )

    def _is_exempt(self, path: str) -> bool:
        """Check if the path is exempt from authentication."""
        return path in self.config.exempt_paths

    async def dispatch(self, request: Request, call_next):
        """Process the request through security checks."""
        path = request.url.path

        # Skip if security is disabled
        if not self.config.enabled:
            return await call_next(request)

        # Check exempt paths
        if self._is_exempt(path):
            return await call_next(request)

        # Extract and validate API key
        api_key = extract_api_key(request)

        if not api_key:
            logger.warning(
                "Missing API key",
                extra={"extra_data": {"path": path, "method": request.method}}
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing API key. Use Authorization: Bearer <api_key>"},
            )

        key_config = self.key_map.get(api_key)

        if not key_config:
            logger.warning(
                "Invalid API key",
                extra={"extra_data": {"path": path, "method": request.method}}
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid API key"},
            )

        # Check rate limit
        if self.config.rate_limiting.enabled:
            limit = key_config.rate_limit
            allowed, headers = self.limiter.is_allowed(api_key, limit)

            if not allowed:
                logger.warning(
                    "Rate limit exceeded",
                    extra={"extra_data": {
                        "key_name": key_config.name,
                        "path": path,
                        "limit": limit,
                    }}
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Rate limit exceeded",
                        "retry_after": int(headers.get("Retry-After", 60)),
                    },
                    headers=headers,
                )
        else:
            headers = {}

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        for key, value in headers.items():
            response.headers[key] = value

        return response
