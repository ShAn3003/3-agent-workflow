from typing import Any, Dict, List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/mnt/xieshan/checkpoints/Qwen2-0.5B-Instruct"

app = FastAPI(title="Local OpenAI-Compatible LLM Server")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)


class ChatRequest(BaseModel):
    model: str = "Qwen2-0.5B-Instruct"
    messages: List[Dict[str, Any]]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    stream: bool = False
    extra_body: Dict[str, Any] = {}


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts).strip()
    return str(content)


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for m in messages:
        role = str(m.get("role", "user"))
        content = _flatten_content(m.get("content", ""))
        out.append({"role": role, "content": content})
    return out


@app.get("/v1/models")
def models() -> Dict[str, Any]:
    return {"data": [{"id": "Qwen2-0.5B-Instruct", "object": "model"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest) -> Dict[str, Any]:
    try:
        chat = _normalize_messages(req.messages)
        prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=max(1, int(req.max_tokens)),
            temperature=max(0.01, float(req.temperature)),
            top_p=float(req.top_p),
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )
        output_ids = gen[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
