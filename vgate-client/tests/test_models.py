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

"""Tests for request/response models."""

from vgate_client.models import (
    ChatCompletion,
    ChatCompletionRequest,
    ChatMessage,
    EmbeddingRequest,
    EmbeddingResponse,
    HealthResponse,
    RateLimitInfo,
    Usage,
)


class TestChatCompletionRequest:
    def test_minimal(self):
        req = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
        )
        assert req.model == "test-model"
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.max_tokens == 256

    def test_custom_params(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="Hello")],
            temperature=0.2,
            top_p=0.5,
            max_tokens=100,
        )
        assert req.temperature == 0.2
        assert req.top_p == 0.5
        assert req.max_tokens == 100

    def test_serialization(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="Hi")],
        )
        data = req.model_dump()
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "Hi"


class TestChatCompletion:
    def test_parse(self, chat_response_payload):
        resp = ChatCompletion.model_validate(chat_response_payload)
        assert resp.id == "chatcmpl-abc12345"
        assert resp.object == "chat.completion"
        assert resp.choices[0].message.content == "Hello! How can I help you?"
        assert resp.usage.total_tokens == 18

    def test_defaults(self):
        resp = ChatCompletion(
            id="x", created=0, model="m",
            choices=[],
        )
        assert resp.usage.prompt_tokens == 0


class TestEmbeddingResponse:
    def test_parse(self, embedding_response_payload):
        resp = EmbeddingResponse.model_validate(embedding_response_payload)
        assert resp.object == "list"
        assert len(resp.data) == 1
        assert len(resp.data[0].embedding) == 1536


class TestEmbeddingRequest:
    def test_basic(self):
        req = EmbeddingRequest(model="m", input="hello world")
        assert req.input == "hello world"


class TestHealthResponse:
    def test_parse(self, health_response_payload):
        resp = HealthResponse.model_validate(health_response_payload)
        assert resp.status == "ok"
        assert resp.version == "0.3.2"


class TestRateLimitInfo:
    def test_defaults(self):
        info = RateLimitInfo()
        assert info.limit is None
        assert info.remaining is None
        assert info.retry_after is None

    def test_with_values(self):
        info = RateLimitInfo(limit=100, remaining=42, reset=1700000000.0, retry_after=5.0)
        assert info.limit == 100
        assert info.remaining == 42


class TestUsage:
    def test_defaults(self):
        u = Usage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0
