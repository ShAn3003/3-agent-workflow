#!/usr/bin/env bash
# 启动 3-Agent 编排 API（默认监听 9000 端口）。
set -euo pipefail
uvicorn src.api:app --host 0.0.0.0 --port 9000
