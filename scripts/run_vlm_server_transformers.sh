#!/usr/bin/env bash
set -euo pipefail
uvicorn src.local_llm_server:app --host 0.0.0.0 --port 8000
