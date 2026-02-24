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

"""Exception classes for the V-Gate client SDK."""

from __future__ import annotations

from typing import Optional


class VGateError(Exception):
    """Base exception for all V-Gate client errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        body: Optional[dict] = None,
    ):
        self.status_code = status_code
        self.body = body
        super().__init__(message)


class AuthenticationError(VGateError):
    """Raised when authentication fails (HTTP 401)."""


class RateLimitError(VGateError):
    """Raised when rate limit is exceeded (HTTP 429).

    Attributes:
        retry_after: Seconds to wait before retrying.
    """

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        **kwargs,
    ):
        self.retry_after = retry_after
        super().__init__(message, **kwargs)


class ServerError(VGateError):
    """Raised on server-side errors (HTTP 5xx)."""


class ConnectionError(VGateError):
    """Raised when the client cannot connect to the V-Gate server."""
