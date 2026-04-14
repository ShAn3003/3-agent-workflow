#!/usr/bin/env bash
# 启动本地 OpenAI 兼容模型服务（默认监听 8000 端口）。
set -euo pipefail
uvicorn src.local_llm_server:app --host 0.0.0.0 --port 8000
