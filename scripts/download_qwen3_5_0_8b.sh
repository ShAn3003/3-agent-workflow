#!/usr/bin/env bash
# 下载 Qwen3.5-0.8B（默认从 Hugging Face），并放到本地目录。
set -euo pipefail

# 可通过环境变量覆盖：
#   MODEL_REPO=Qwen/Qwen3.5-0.8B
#   TARGET_DIR=/path/to/your/models/Qwen3.5-0.8B
# TODO: 将 TARGET_DIR 改成你机器上的真实模型目录（建议使用绝对路径）。
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.5-0.8B}"
TARGET_DIR="${TARGET_DIR:-./models/Qwen3.5-0.8B}"
export MODEL_REPO TARGET_DIR

echo "[INFO] model repo: ${MODEL_REPO}"
echo "[INFO] target dir: ${TARGET_DIR}"

mkdir -p "${TARGET_DIR}"

python3 - <<'PY'
import os
import sys

repo = os.environ.get("MODEL_REPO", "Qwen/Qwen3.5-0.8B")
# TODO: 将 TARGET_DIR 改成你机器上的真实模型目录（建议使用绝对路径）。
target = os.environ.get("TARGET_DIR", "./models/Qwen3.5-0.8B")

try:
    from huggingface_hub import snapshot_download
except Exception:
    print(
        "[ERROR] missing huggingface_hub. Run: pip install -U huggingface_hub",
        file=sys.stderr,
    )
    sys.exit(1)

snapshot_download(
    repo_id=repo,
    local_dir=target,
    local_dir_use_symlinks=False,
    resume_download=True,
)
print(f"[OK] downloaded to: {target}")
PY
