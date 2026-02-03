"""
Tests for V-Gate security module.

Tests cover:
- API key authentication
- Rate limiting with sliding window
- Exempt paths
- X-RateLimit-* headers
- Error responses (401, 429)
"""
import time
from unittest.mock import MagicMock, AsyncMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from vgate.config import SecurityConfig, APIKeyConfig, RateLimitConfig
from vgate.security import RateLimiter, extract_api_key, SecurityMiddleware


class TestRateLimiter:
    """Tests for the RateLimiter class."""

    def test_allows_requests_under_limit(self):
        """Requests under the limit should be allowed."""
        limiter = RateLimiter(window_seconds=60)

        for i in range(5):
            allowed, headers = limiter.is_allowed("test-key", limit=10)
            assert allowed is True
            assert headers["X-RateLimit-Limit"] == "10"
            assert headers["X-RateLimit-Remaining"] == str(10 - i - 1)

    def test_blocks_requests_over_limit(self):
        """Requests over the limit should be blocked."""
        limiter = RateLimiter(window_seconds=60)

        # Use up all requests
        for _ in range(5):
            limiter.is_allowed("test-key", limit=5)

        # Next request should be blocked
        allowed, headers = limiter.is_allowed("test-key", limit=5)
        assert allowed is False
        assert headers["X-RateLimit-Remaining"] == "0"
        assert "Retry-After" in headers

    def test_separate_limits_per_key(self):
        """Different keys should have separate limits."""
        limiter = RateLimiter(window_seconds=60)

        # Use up key1's limit
        for _ in range(3):
            limiter.is_allowed("key1", limit=3)

        # key1 should be blocked
        allowed1, _ = limiter.is_allowed("key1", limit=3)
        assert allowed1 is False

        # key2 should still be allowed
        allowed2, _ = limiter.is_allowed("key2", limit=3)
        assert allowed2 is True

    def test_window_reset(self):
        """Requests should be allowed after window resets."""
        limiter = RateLimiter(window_seconds=1)  # 1 second window

        # Use up the limit
        for _ in range(3):
            limiter.is_allowed("test-key", limit=3)

        # Should be blocked
        allowed, _ = limiter.is_allowed("test-key", limit=3)
        assert allowed is False

        # Wait for window to pass
        time.sleep(1.1)

        # Should be allowed again
        allowed, _ = limiter.is_allowed("test-key", limit=3)
        assert allowed is True

    def test_rate_limit_headers(self):
        """Rate limit headers should be correct."""
        limiter = RateLimiter(window_seconds=60)

        allowed, headers = limiter.is_allowed("test-key", limit=100)

        assert "X-RateLimit-Limit" in headers
        assert "X-RateLimit-Remaining" in headers
        assert "X-RateLimit-Reset" in headers
        assert int(headers["X-RateLimit-Reset"]) > int(time.time())

    def test_get_usage(self):
        """get_usage should return current stats."""
        limiter = RateLimiter(window_seconds=60)

        # Make some requests
        for _ in range(5):
            limiter.is_allowed("test-key", limit=10)

        usage = limiter.get_usage("test-key")
        assert usage["current_requests"] == 5
        assert usage["window_seconds"] == 60


class TestExtractApiKey:
    """Tests for the extract_api_key function."""

    def test_extracts_bearer_token(self):
        """Should extract API key from Bearer token."""
        request = MagicMock(spec=Request)
        request.headers = {"Authorization": "Bearer sk-test-key-123"}

        key = extract_api_key(request)
        assert key == "sk-test-key-123"

    def test_returns_none_for_missing_header(self):
        """Should return None if Authorization header is missing."""
        request = MagicMock(spec=Request)
        request.headers = {}

        key = extract_api_key(request)
        assert key is None

    def test_returns_none_for_invalid_format(self):
        """Should return None for invalid Authorization format."""
        request = MagicMock(spec=Request)

        # No "Bearer" prefix
        request.headers = {"Authorization": "sk-test-key-123"}
        assert extract_api_key(request) is None

        # Wrong prefix
        request.headers = {"Authorization": "Basic sk-test-key-123"}
        assert extract_api_key(request) is None

        # Too many parts
        request.headers = {"Authorization": "Bearer sk-test extra"}
        assert extract_api_key(request) is None

    def test_case_insensitive_bearer(self):
        """Bearer prefix should be case insensitive."""
        request = MagicMock(spec=Request)

        request.headers = {"Authorization": "bearer sk-test-key"}
        assert extract_api_key(request) == "sk-test-key"

        request.headers = {"Authorization": "BEARER sk-test-key"}
        assert extract_api_key(request) == "sk-test-key"


class TestSecurityMiddleware:
    """Tests for the SecurityMiddleware class."""

    @pytest.fixture
    def security_config(self):
        """Create a test security configuration."""
        return SecurityConfig(
            enabled=True,
            api_keys=[
                APIKeyConfig(key="sk-valid-key", name="test", rate_limit=10),
                APIKeyConfig(key="sk-limited-key", name="limited", rate_limit=2),
            ],
            rate_limiting=RateLimitConfig(
                enabled=True,
                default_limit=5,
                window_seconds=60,
            ),
            exempt_paths=["/health", "/metrics"],
        )

    @pytest.fixture
    def app_with_security(self, security_config):
        """Create a test FastAPI app with security middleware."""
        app = FastAPI()
        app.add_middleware(SecurityMiddleware, config=security_config)

        @app.get("/health")
        def health():
            return {"status": "ok"}

        @app.get("/metrics")
        def metrics():
            return {"metrics": "data"}

        @app.get("/api/test")
        def api_test():
            return {"message": "success"}

        @app.post("/v1/chat/completions")
        def chat():
            return {"response": "hello"}

        return app

    def test_exempt_paths_allowed_without_auth(self, app_with_security):
        """Exempt paths should be accessible without authentication."""
        client = TestClient(app_with_security)

        response = client.get("/health")
        assert response.status_code == 200

        response = client.get("/metrics")
        assert response.status_code == 200

    def test_protected_path_requires_auth(self, app_with_security):
        """Protected paths should require authentication."""
        client = TestClient(app_with_security)

        response = client.get("/api/test")
        assert response.status_code == 401
        assert "Missing API key" in response.json()["detail"]

    def test_valid_api_key_allowed(self, app_with_security):
        """Valid API key should be allowed."""
        client = TestClient(app_with_security)

        response = client.get(
            "/api/test",
            headers={"Authorization": "Bearer sk-valid-key"}
        )
        assert response.status_code == 200
        assert response.json()["message"] == "success"

    def test_invalid_api_key_rejected(self, app_with_security):
        """Invalid API key should be rejected."""
        client = TestClient(app_with_security)

        response = client.get(
            "/api/test",
            headers={"Authorization": "Bearer sk-invalid-key"}
        )
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]

    def test_rate_limit_enforced(self, app_with_security):
        """Rate limit should be enforced."""
        client = TestClient(app_with_security)
        headers = {"Authorization": "Bearer sk-limited-key"}

        # First 2 requests should succeed (rate_limit=2)
        for _ in range(2):
            response = client.get("/api/test", headers=headers)
            assert response.status_code == 200

        # Third request should be rate limited
        response = client.get("/api/test", headers=headers)
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]
        assert "retry_after" in response.json()

    def test_rate_limit_headers_present(self, app_with_security):
        """Rate limit headers should be present in responses."""
        client = TestClient(app_with_security)
        headers = {"Authorization": "Bearer sk-valid-key"}

        response = client.get("/api/test", headers=headers)
        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    def test_security_disabled(self, security_config):
        """When security is disabled, all requests should be allowed."""
        security_config.enabled = False

        app = FastAPI()
        app.add_middleware(SecurityMiddleware, config=security_config)

        @app.get("/api/test")
        def api_test():
            return {"message": "success"}

        client = TestClient(app)
        response = client.get("/api/test")
        assert response.status_code == 200

    def test_rate_limiting_disabled(self):
        """When rate limiting is disabled, no rate limits should apply."""
        config = SecurityConfig(
            enabled=True,
            api_keys=[
                APIKeyConfig(key="sk-test", name="test", rate_limit=1),
            ],
            rate_limiting=RateLimitConfig(enabled=False),
            exempt_paths=[],
        )

        app = FastAPI()
        app.add_middleware(SecurityMiddleware, config=config)

        @app.get("/api/test")
        def api_test():
            return {"message": "success"}

        client = TestClient(app)
        headers = {"Authorization": "Bearer sk-test"}

        # Should be able to make many requests even though rate_limit=1
        for _ in range(10):
            response = client.get("/api/test", headers=headers)
            assert response.status_code == 200


class TestSecurityConfigModels:
    """Tests for security configuration models."""

    def test_api_key_config_defaults(self):
        """Test APIKeyConfig default values."""
        config = APIKeyConfig(key="sk-test", name="test")
        assert config.rate_limit == 60

    def test_rate_limit_config_defaults(self):
        """Test RateLimitConfig default values."""
        config = RateLimitConfig()
        assert config.enabled is True
        assert config.default_limit == 60
        assert config.window_seconds == 60

    def test_security_config_defaults(self):
        """Test SecurityConfig default values."""
        config = SecurityConfig()
        assert config.enabled is False
        assert config.api_keys == []
        assert config.exempt_paths == ["/health", "/metrics"]

    def test_security_config_with_api_keys(self):
        """Test SecurityConfig with API keys."""
        config = SecurityConfig(
            enabled=True,
            api_keys=[
                APIKeyConfig(key="sk-1", name="key1", rate_limit=100),
                APIKeyConfig(key="sk-2", name="key2", rate_limit=50),
            ],
        )
        assert len(config.api_keys) == 2
        assert config.api_keys[0].key == "sk-1"
        assert config.api_keys[1].rate_limit == 50
