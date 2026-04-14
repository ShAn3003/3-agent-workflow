"""对外暴露的编排服务 API。"""

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .orchestrator import LLMEndpoint, run_three_agents

app = FastAPI(title="LLM Agent Lab")


class SolveRequest(BaseModel):
    """`/solve` 请求体定义。"""

    # 调用链路 ID。为空时由编排器自动生成 UUID。
    trace_id: Optional[str] = None
    # 下游模型名，需与本地 OpenAI 兼容服务中的模型 ID 一致。
    model: str = os.environ.get("MODEL_ID", "Qwen3.5-0.8B")
    # 标准 Chat Completions 消息数组。
    messages: List[Dict[str, Any]]
    # 生成参数，保持与 OpenAI 风格接口一致。
    gen: Dict[str, Any] = Field(default_factory=lambda: {
        "max_tokens": 800,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "stream": False,
    })


@app.get("/health")
async def health() -> Dict[str, str]:
    """健康检查接口。"""

    return {"status": "ok"}


@app.post("/solve")
async def solve(req: SolveRequest) -> Dict[str, Any]:
    """执行 3-Agent 串行流程并返回聚合结果。"""

    # 这里固定连接本地 8000 端口的 OpenAI 兼容后端。
    endpoint = LLMEndpoint(base_url="http://localhost:8000/v1", api_key="EMPTY")
    try:
        return await run_three_agents(endpoint, req.model, req.messages, req.gen, req.trace_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
