# Copyright 2025 the V-Gate authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for exception classes."""

from vgate_client.exceptions import (
    AuthenticationError,
    ConnectionError,
    RateLimitError,
    ServerError,
    VGateError,
)


class TestVGateError:
    def test_base_error(self):
        err = VGateError("something went wrong", status_code=400)
        assert str(err) == "something went wrong"
        assert err.status_code == 400
        assert err.body is None

    def test_with_body(self):
        body = {"detail": "bad request"}
        err = VGateError("bad", status_code=400, body=body)
        assert err.body == body


class TestAuthenticationError:
    def test_is_vgate_error(self):
        err = AuthenticationError("Invalid API key", status_code=401)
        assert isinstance(err, VGateError)
        assert err.status_code == 401


class TestRateLimitError:
    def test_retry_after(self):
        err = RateLimitError("Rate limit exceeded", retry_after=30.0, status_code=429)
        assert isinstance(err, VGateError)
        assert err.retry_after == 30.0
        assert err.status_code == 429

    def test_no_retry_after(self):
        err = RateLimitError("Rate limit exceeded", status_code=429)
        assert err.retry_after is None


class TestServerError:
    def test_basic(self):
        err = ServerError("Internal server error", status_code=500)
        assert isinstance(err, VGateError)
        assert err.status_code == 500


class TestConnectionError:
    def test_basic(self):
        err = ConnectionError("Cannot connect to server")
        assert isinstance(err, VGateError)
        assert err.status_code is None


class TestExceptionHierarchy:
    def test_all_inherit_vgate_error(self):
        for cls in (AuthenticationError, RateLimitError, ServerError, ConnectionError):
            assert issubclass(cls, VGateError)

    def test_catchable_as_base(self):
        try:
            raise AuthenticationError("test", status_code=401)
        except VGateError as e:
            assert e.status_code == 401
