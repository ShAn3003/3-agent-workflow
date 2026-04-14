"""本地 OpenAI 兼容推理服务。"""

from typing import Any, Dict, List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# 默认模型路径。若模型不在该目录，请按 README 指引修改。
MODEL_PATH = "/mnt/xieshan/checkpoints/Qwen2-0.5B-Instruct"

app = FastAPI(title="Local OpenAI-Compatible LLM Server")

# 启动时加载 tokenizer / model，避免每次请求重复初始化。
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)


class ChatRequest(BaseModel):
    """`/v1/chat/completions` 请求体。"""

    model: str = "Qwen2-0.5B-Instruct"
    messages: List[Dict[str, Any]]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    stream: bool = False
    # 与 OpenAI 兼容字段对齐，当前实现未深度使用。
    extra_body: Dict[str, Any] = Field(default_factory=dict)


def _flatten_content(content: Any) -> str:
    """将字符串或多模态 content 结构扁平化为纯文本。"""

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
    """规范化消息列表，统一 role/content 为字符串。"""

    out: List[Dict[str, str]] = []
    for m in messages:
        role = str(m.get("role", "user"))
        content = _flatten_content(m.get("content", ""))
        out.append({"role": role, "content": content})
    return out


@app.get("/v1/models")
def models() -> Dict[str, Any]:
    """模型列表接口（最小实现）。"""

    return {"data": [{"id": "Qwen2-0.5B-Instruct", "object": "model"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest) -> Dict[str, Any]:
    """OpenAI 风格聊天补全接口（非流式）。"""

    try:
        chat = _normalize_messages(req.messages)
        prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        # 当前最小实现固定为采样模式；若 temperature 越低，结果越稳定。
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
