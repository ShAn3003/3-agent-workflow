import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .logutil import append_jsonl


@dataclass
class LLMEndpoint:
    base_url: str
    api_key: str = "EMPTY"
    timeout_s: float = 180.0
    max_retries: int = 2


async def _chat(
    client: httpx.AsyncClient,
    endpoint: LLMEndpoint,
    model: str,
    messages: List[Dict[str, Any]],
    gen: Dict[str, Any],
) -> Tuple[str, int, int]:
    url = f"{endpoint.base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": int(gen.get("max_tokens", 800)),
        "temperature": float(gen.get("temperature", 0.7)),
        "top_p": float(gen.get("top_p", 0.8)),
        "stream": bool(gen.get("stream", False)),
        "extra_body": {"top_k": int(gen.get("top_k", 20))},
    }

    last_err: Optional[Exception] = None
    for _ in range(endpoint.max_retries + 1):
        try:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            text = (data["choices"][0]["message"]["content"] or "").strip()
            in_len = len(str(messages))
            out_len = len(text)
            return text, in_len, out_len
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.5)
    raise RuntimeError(f"chat_completions failed: {last_err!r}")


def solver_system() -> str:
    return "你是Agent-1（解题）。清晰作答，不确定请标注。"


def reviewer_system() -> str:
    return (
        "你是Agent-2（审稿）。输出4段："
        "1) 错误/不一致点（至少3条，若无则解释）"
        "2) 幻觉风险"
        "3) 改进建议"
        "4) 改进版草稿（可选）"
    )


def synthesizer_system() -> str:
    return "你是Agent-3（综合定稿）。逐条吸收审稿意见，给最终答案并标注不确定性。"


async def run_three_agents(
    endpoint: LLMEndpoint,
    model: str,
    user_messages: List[Dict[str, Any]],
    gen: Dict[str, Any],
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    tid = trace_id or str(uuid.uuid4())
    t0 = time.time()

    err = None
    a1 = a2 = final = ""
    a1_meta = a2_meta = a3_meta = {"latency_s": 0.0, "input_len": 0, "output_len": 0}

    try:
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {endpoint.api_key}"},
            timeout=endpoint.timeout_s,
            trust_env=False,
        ) as client:
            s = time.time()
            a1_messages = [{"role": "system", "content": solver_system()}, *user_messages]
            a1, i1, o1 = await _chat(client, endpoint, model, a1_messages, gen)
            a1_meta = {"latency_s": round(time.time() - s, 3), "input_len": i1, "output_len": o1}

            s = time.time()
            a2_messages = [
                {"role": "system", "content": reviewer_system()},
                *user_messages,
                {"role": "assistant", "content": a1},
                {"role": "user", "content": "请按要求审阅上面的回答。"},
            ]
            a2, i2, o2 = await _chat(client, endpoint, model, a2_messages, gen)
            a2_meta = {"latency_s": round(time.time() - s, 3), "input_len": i2, "output_len": o2}

            s = time.time()
            a3_messages = [
                {"role": "system", "content": synthesizer_system()},
                *user_messages,
                {"role": "assistant", "content": f"[Agent-1]\n{a1}"},
                {"role": "assistant", "content": f"[Agent-2]\n{a2}"},
                {"role": "user", "content": "请给出最终答案。"},
            ]
            final, i3, o3 = await _chat(client, endpoint, model, a3_messages, gen)
            a3_meta = {"latency_s": round(time.time() - s, 3), "input_len": i3, "output_len": o3}

    except Exception as e:
        err = repr(e)

    total = round(time.time() - t0, 3)
    result = {
        "trace_id": tid,
        "model": model,
        "latency_s": total,
        "agent1": a1,
        "agent2": a2,
        "final": final,
        "agent_metrics": {"agent1": a1_meta, "agent2": a2_meta, "agent3": a3_meta},
        "error": err,
    }

    append_jsonl(result)
    if err:
        raise RuntimeError(err)
    return result
