from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys
import time

# Add the vgate directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'vgate')))

# Import the actual VGateEngine
from engine import VGateEngine

app = FastAPI(
    title="V-Gate AI Model Serving Gateway",
    description="A high-performance AI model serving gateway for various models.",
    version="0.1.0",
)

# Initialize the VGateEngine
# Model name should ideally come from configuration in a later stage
engine = VGateEngine(model_name="Qwen/Qwen2.5-1.5B-Instruct-AWQ")

# Request models for OpenAI-like API
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    # Add other fields as needed for OpenAI compatibility (e.g., temperature, max_tokens)

class EmbeddingRequest(BaseModel):
    model: str
    input: str
    # Add other fields as needed

# Helper function to convert messages to a prompt string
def messages_to_prompt(messages: list) -> str:
    prompt_parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        prompt_parts.append(f"{role.capitalize()}: {content}")
    return "\n".join(prompt_parts) + "\nAssistant:"


@app.get("/health", summary="Health Check")
async def health_check():
    """
    Returns the health status of the V-Gate service.
    """
    return {"status": "ok"}

@app.post("/v1/chat/completions", summary="Create Chat Completion")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Generates a chat completion response from the specified model.
    """
    try:
        # Convert messages to a single prompt string for the engine
        prompt = messages_to_prompt(request.messages)
        response = engine.chat_completions(prompt)
        
        # Adapt engine's response to OpenAI-like format
        return {
            "id": "chatcmpl-" + str(hash(prompt)), # Simple unique ID for mock
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response["text"],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": response.get("prompt_tokens", 0), # Assuming engine can return this
                "completion_tokens": response.get("total_tokens", 0),
                "total_tokens": response.get("total_tokens", 0) + response.get("prompt_tokens", 0),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/embeddings", summary="Create Embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """
    Generates embeddings for the given input text from the specified model.
    """
    try:
        response = engine.embeddings(request.input)
        
        # Adapt engine's response to OpenAI-like format
        return {
            "object": "list",
            "data": response["data"],
            "model": request.model,
            "usage": response["usage"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
