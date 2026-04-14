from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .orchestrator import LLMEndpoint, run_three_agents

app = FastAPI(title="LLM Agent Lab")


class SolveRequest(BaseModel):
    trace_id: Optional[str] = None
    model: str = "Qwen2-0.5B-Instruct"
    messages: List[Dict[str, Any]]
    gen: Dict[str, Any] = Field(default_factory=lambda: {
        "max_tokens": 800,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "stream": False,
    })


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/solve")
async def solve(req: SolveRequest):
    endpoint = LLMEndpoint(base_url="http://localhost:8000/v1", api_key="EMPTY")
    try:
        return await run_three_agents(endpoint, req.model, req.messages, req.gen, req.trace_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
