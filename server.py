import json
import os
import re
from typing import List, Optional
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from llama_cpp import Llama
from pydantic import BaseModel

# =========================
# Server & Authentication
# =========================

app = FastAPI(title="Local Qwen3 LLM Server")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

EXPECTED_API_KEY = os.getenv("LLM_API_KEY", "your-secret-api-key")


def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    if not api_key or api_key != EXPECTED_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing API Key",
        )
    return api_key


# =========================
# Model Setup
# =========================

MODEL_PATH = r"D:\A ai projects\llm\ai_models\Qwen3-4B-Q5_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=8192,
    n_threads=8,
    n_batch=512,
    verbose=True,
)

# =========================
# Schemas
# =========================


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.8
    stream: Optional[bool] = False  # Set default to False for non-streaming


# UPDATE: Define schema returning both thinking and response
class ChatResponse(BaseModel):
    thinking: str
    response: str


# =========================
# Endpoint
# =========================


@app.post("/chat")
async def chat_endpoint(
    request: ChatRequest, api_key: str = Depends(verify_api_key)
):
    formatted_messages = [msg.model_dump() for msg in request.messages]

    stream = llm.create_chat_completion(
        messages=formatted_messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=request.stream,
    )

    if request.stream:

        def event_generator():
            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {})
                text = delta.get("content", "")
                if text:
                    yield f"data: {json.dumps({'text': text})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(), media_type="text/event-stream"
        )
    else:
        full_text = stream["choices"][0]["message"]["content"]

        thinking_content = ""
        clean_response = full_text

        # Safely extract <think> tags if present
        think_match = re.search(r"<think>(.*?)</think>", full_text, re.DOTALL)
        if think_match:
            thinking_content = think_match.group(1).strip()
            clean_response = re.sub(
                r"<think>.*?</think>", "", full_text, flags=re.DOTALL
            ).strip()
        else:
            # Handle cases where model starts with <think> but forgets closing tag
            if full_text.startswith("<think>"):
                parts = full_text.split("<think>", 1)
                thinking_content = parts[1].strip()
                clean_response = ""

        return {"thinking": thinking_content, "response": clean_response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)