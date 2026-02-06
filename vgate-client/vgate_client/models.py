"""Request and response models for V-Gate API, compatible with OpenAI format."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ── Request Models ──────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256


class EmbeddingRequest(BaseModel):
    model: str
    input: str


# ── Response Models ─────────────────────────────────────────────────────────


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ResponseMessage(BaseModel):
    role: str
    content: str


class Choice(BaseModel):
    index: int
    message: ResponseMessage
    finish_reason: Optional[str] = None


class ChatCompletion(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: Usage = Field(default_factory=Usage)


class HealthResponse(BaseModel):
    status: str
    version: str


class RateLimitInfo(BaseModel):
    """Rate limit information extracted from response headers."""
    limit: Optional[int] = None
    remaining: Optional[int] = None
    reset: Optional[float] = None
    retry_after: Optional[float] = None
