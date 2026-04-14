# llm-agent-lab

Minimal 3-agent workflow:
- Agent-1: solve
- Agent-2: review
- Agent-3: synthesize final answer

The repo exposes one API:
- `POST /solve` on port `9000`

It depends on a local OpenAI-compatible model server on port `8000`.

## Repo Structure

```text
llm-agent-lab/
  src/
    api.py                 # /health and /solve
    orchestrator.py        # 3-agent serial orchestration
    local_llm_server.py    # local OpenAI-compatible model server
    logutil.py             # JSONL logging
  scripts/
    run_vlm_server_transformers.sh
    run_orchestrator.sh
    smoke_test_text.py
    smoke_test_vlm.py
  outputs/logs/
```

## Prerequisites

1. Python env with packages from `requirements.txt`.
2. Local model files (default):
   - `/mnt/xieshan/checkpoints/Qwen2-0.5B-Instruct`
3. Optional GPU pinning (recommended on shared machines):
   - `CUDA_VISIBLE_DEVICES=7`

If your model path is different, update `MODEL_PATH` in `src/local_llm_server.py`.

## Install (if needed)

If your environment already has `fastapi`, `uvicorn`, `httpx`, `transformers`, `torch`, `requests`, skip this.

```bash
cd /mnt/xieshan/program/llm-agent-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Run

Open 2 terminals.

### Terminal 1: start local model server (`:8000`)

```bash
cd /mnt/xieshan/program/llm-agent-lab
CUDA_VISIBLE_DEVICES=7 uvicorn src.local_llm_server:app --host 0.0.0.0 --port 8000
```

### Terminal 2: start orchestrator (`:9000`)

```bash
cd /mnt/xieshan/program/llm-agent-lab
uvicorn src.api:app --host 0.0.0.0 --port 9000
```

## Smoke Tests

```bash
cd /mnt/xieshan/program/llm-agent-lab
python scripts/smoke_test_text.py
python scripts/smoke_test_vlm.py
```

Expected:
- HTTP status `200`
- JSON contains: `trace_id`, `agent1`, `agent2`, `final`, `agent_metrics`, `error`

## API Usage

### Health

```bash
curl --noproxy '*' -sS http://127.0.0.1:9000/health
```

### Solve

```bash
curl --noproxy '*' -sS http://127.0.0.1:9000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role":"user","content":"Explain ReAct briefly."}]
  }'
```

## Logs

JSONL logs are written to:
- `outputs/logs/run-YYYYMMDD.jsonl`

Each record includes:
- `trace_id`
- per-agent latency/input/output lengths
- `error` (null when successful)

## Important Notes

1. Current `src/local_llm_server.py` uses a text model (`Qwen2-0.5B-Instruct`).
2. `smoke_test_vlm.py` sends image-structured payload, but the local server currently only consumes text fields in `content`.
3. If you need true image understanding, replace the local model server with a real VLM backend.

## Troubleshooting

1. `502` from `/solve`:
   - confirm model server is running on `127.0.0.1:8000`
2. Requests hang/time out locally:
   - proxy env may interfere; use `--noproxy '*'` in curl
3. GPU contention:
   - run with `CUDA_VISIBLE_DEVICES=7` (or your dedicated GPU id)
4. Port already in use:
   - check with `ss -ltnp | rg ':8000|:9000'`
