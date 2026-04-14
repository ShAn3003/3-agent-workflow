#!/usr/bin/env bash
set -euo pipefail
uvicorn src.api:app --host 0.0.0.0 --port 9000
