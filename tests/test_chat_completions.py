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

"""
Tests for /v1/chat/completions request validation.

ChatCompletionRequest.messages used to be a bare `list`, so a malformed
message (missing "role"/"content", or not a dict at all) would pass
validation and then blow up as an unhandled 500 in messages_to_prompt()
instead of a clean 422.
"""

import pytest

from vgate.config import reset_config


@pytest.fixture(autouse=True)
def _reset():
    reset_config()
    yield
    reset_config()


class TestChatCompletionValidation:
    @pytest.mark.asyncio
    async def test_well_formed_messages_succeed(self):
        from httpx import ASGITransport, AsyncClient
        from main import app, lifespan

        async with lifespan(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/v1/chat/completions", json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                })

            assert response.status_code == 200
            assert "choices" in response.json()

    @pytest.mark.asyncio
    async def test_message_missing_content_returns_422_not_500(self):
        from httpx import ASGITransport, AsyncClient
        from main import app, lifespan

        async with lifespan(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/v1/chat/completions", json={
                    "model": "test-model",
                    "messages": [{"role": "user"}],
                })

            assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_message_not_an_object_returns_422_not_500(self):
        from httpx import ASGITransport, AsyncClient
        from main import app, lifespan

        async with lifespan(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/v1/chat/completions", json={
                    "model": "test-model",
                    "messages": ["just a string, not a message object"],
                })

            assert response.status_code == 422
