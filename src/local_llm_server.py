"""本地 OpenAI 兼容推理服务。"""

import base64
import inspect
import os
from io import BytesIO
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import requests
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# TODO: 请在部署环境中通过 MODEL_PATH 显式设置真实模型目录，避免依赖示例默认值。
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/Qwen3.5-0.8B")
MODEL_ID = os.environ.get("MODEL_ID", os.path.basename(MODEL_PATH.rstrip("/")) or "local-model")

app = FastAPI(title="Local OpenAI-Compatible LLM Server")

# 启动时加载 tokenizer / model，避免每次请求重复初始化。
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
try:
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
except Exception:
    processor = None

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)


class ChatRequest(BaseModel):
    """`/v1/chat/completions` 请求体。"""

    model: str = MODEL_ID
    messages: List[Dict[str, Any]]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    stream: bool = False
    # 与 OpenAI 兼容字段对齐，当前实现未深度使用。
    extra_body: Dict[str, Any] = Field(default_factory=dict)


def _load_image_from_url(url: str) -> Image.Image:
    """从 http(s)/data/file/local path 读取图片。"""

    if not isinstance(url, str) or not url.strip():
        raise ValueError("invalid image url")

    u = url.strip()
    if u.startswith("data:image/"):
        try:
            b64 = u.split(",", 1)[1]
            raw = base64.b64decode(b64)
            return Image.open(BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise ValueError(f"bad data URL image: {e}") from e

    if u.startswith("http://") or u.startswith("https://"):
        try:
            r = requests.get(u, timeout=20)
            r.raise_for_status()
            return Image.open(BytesIO(r.content)).convert("RGB")
        except Exception as e:
            raise ValueError(f"failed to fetch image URL: {e}") from e

    if u.startswith("file://"):
        path = urlparse(u).path
    else:
        path = u
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception as e:
        raise ValueError(f"failed to load local image '{path}': {e}") from e


def _extract_image_ref(item: Dict[str, Any]) -> str:
    """从多种 OpenAI 风格字段中提取图片引用。"""

    t = item.get("type")
    if t == "image_url":
        raw = item.get("image_url")
        if isinstance(raw, str):
            return raw
        if isinstance(raw, dict):
            return str(raw.get("url", ""))
    if t == "image":
        if isinstance(item.get("image"), str):
            return str(item["image"])
        raw = item.get("image_url")
        if isinstance(raw, str):
            return raw
        if isinstance(raw, dict):
            return str(raw.get("url", ""))
    return ""


def _normalize_messages(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], List[Image.Image]]:
    """规范化消息列表；保留文本并提取图片。"""

    out: List[Dict[str, str]] = []
    images: List[Image.Image] = []

    for m in messages:
        role = str(m.get("role", "user"))
        content = m.get("content", "")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if not isinstance(item, dict):
                    parts.append(str(item))
                    continue
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                    continue
                image_ref = _extract_image_ref(item)
                if image_ref:
                    images.append(_load_image_from_url(image_ref))
                    parts.append("<image>")
            out.append({"role": role, "content": "\n".join(p for p in parts if p).strip()})
            continue

        out.append({"role": role, "content": str(content)})

    return out, images


def _to_model_device(batch: Dict[str, Any]) -> Dict[str, Any]:
    """将 processor/tokenizer 产物迁移到模型设备。"""

    out: Dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(model.device) if hasattr(v, "to") else v
    return out


def _build_text_inputs(chat: List[Dict[str, str]]) -> Dict[str, Any]:
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return tokenizer(prompt, return_tensors="pt").to(model.device)


def _build_multimodal_inputs(chat: List[Dict[str, str]], images: List[Image.Image]) -> Dict[str, Any]:
    if processor is None:
        raise RuntimeError("processor is unavailable; multimodal inputs are not supported")
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    try:
        mm = processor(text=[prompt], images=images, return_tensors="pt", padding=True)
    except TypeError:
        mm = processor(text=[prompt], images=images, return_tensors="pt")
    return _to_model_device(mm)


def _processor_supports_images() -> bool:
    if processor is None:
        return False
    try:
        return "images" in inspect.signature(processor.__call__).parameters
    except Exception:
        return False


def _decode_generation(generated_ids: Any, prompt_len: int) -> str:
    output_ids = generated_ids[0][prompt_len:]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def _should_fallback_to_text(exc: Exception) -> bool:
    msg = str(exc)
    return "model_kwargs" in msg and "not used by the model" in msg


@app.get("/v1/models")
def models() -> Dict[str, Any]:
    """模型列表接口（最小实现）。"""

    return {"data": [{"id": MODEL_ID, "object": "model"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest) -> Dict[str, Any]:
    """OpenAI 风格聊天补全接口（非流式）。"""

    try:
        chat, images = _normalize_messages(req.messages)
        attempted_mm = bool(images and _processor_supports_images())
        if attempted_mm:
            inputs = _build_multimodal_inputs(chat, images)
        else:
            inputs = _build_text_inputs(chat)
        # 当前最小实现固定为采样模式；若 temperature 越低，结果越稳定。
        try:
            gen = model.generate(
                **inputs,
                max_new_tokens=max(1, int(req.max_tokens)),
                temperature=max(0.01, float(req.temperature)),
                top_p=float(req.top_p),
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
            )
        except Exception as e:
            if not attempted_mm or not _should_fallback_to_text(e):
                raise
            # 某些“文本模型 + processor”组合会返回视觉字段但模型不接收，自动回退文本路径。
            inputs = _build_text_inputs(chat)
            gen = model.generate(
                **inputs,
                max_new_tokens=max(1, int(req.max_tokens)),
                temperature=max(0.01, float(req.temperature)),
                top_p=float(req.top_p),
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0
        text = _decode_generation(gen, prompt_len)
        return {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
