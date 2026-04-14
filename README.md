# llm-agent-lab

`llm-agent-lab` 是一个最小可运行的 3-Agent 串行编排实验仓库：

1. Agent-1：先解题，输出初稿  
2. Agent-2：做审稿，指出问题并给改进建议  
3. Agent-3：综合前两者，给最终答案  

项目对外只暴露一个编排接口：`POST /solve`（默认端口 `9000`），并依赖一个本地 OpenAI 兼容模型服务（默认端口 `8000`）。

## 代码结构

```text
llm-agent-lab/
  src/
    api.py                      # 对外 API：/health、/solve
    orchestrator.py             # 3-Agent 串行编排核心
    local_llm_server.py         # 本地 OpenAI 兼容模型服务
    logutil.py                  # JSONL 日志工具
  scripts/
    run_vlm_server_transformers.sh
    run_orchestrator.sh
    download_qwen3_5_0_8b.sh
    cli_solve.py
    smoke_test_text.py
    smoke_test_vlm.py
  outputs/logs/                 # 运行日志输出目录
  requirements.txt
  README.md
```

## 环境要求

1. Python 3.10+（建议）  
2. 安装 `requirements.txt` 中依赖  
3. 本地模型目录（默认示例）：`./models/Qwen3.5-0.8B`  

如果你的模型路径不同，请通过环境变量 `MODEL_PATH` 指定。  
TODO: 替换成你们团队统一的模型存储路径规范，并写入部署文档。

## 安装依赖

```bash
cd /path/to/llm-agent-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

如果需要下载模型脚本依赖，请额外安装：

```bash
pip install -U huggingface_hub
```

## 下载 Qwen3.5-0.8B

默认下载仓库 `Qwen/Qwen3.5-0.8B` 到 `./models/Qwen3.5-0.8B`：

```bash
cd /path/to/llm-agent-lab
bash scripts/download_qwen3_5_0_8b.sh
```

可通过环境变量覆盖：

```bash
MODEL_REPO=Qwen/Qwen3.5-0.8B \
TARGET_DIR=/path/to/your/models/Qwen3.5-0.8B \
bash scripts/download_qwen3_5_0_8b.sh
```

## 启动方式

需要两个终端分别启动模型服务和编排服务。

### 终端 1：启动本地模型服务（8000）

```bash
cd /path/to/llm-agent-lab
CUDA_VISIBLE_DEVICES=7 uvicorn src.local_llm_server:app --host 0.0.0.0 --port 8000
```

### 终端 2：启动编排服务（9000）

```bash
cd /path/to/llm-agent-lab
uvicorn src.api:app --host 0.0.0.0 --port 9000
```

也可以使用脚本启动：

```bash
bash scripts/run_vlm_server_transformers.sh
bash scripts/run_orchestrator.sh
```

## 快速验证

```bash
cd /path/to/llm-agent-lab
python scripts/smoke_test_text.py
python scripts/smoke_test_vlm.py
```

预期结果：

1. HTTP 状态码为 `200`  
2. 返回 JSON 包含 `trace_id`、`agent1`、`agent2`、`final`、`agent_metrics`、`error` 字段

## CLI 模式（输入/输出）

在服务启动后，可使用命令行模式直接提问。

单次提问：

```bash
cd /path/to/llm-agent-lab
python scripts/cli_solve.py --prompt "请解释 ReAct，并给一个示例。"
```

交互模式：

```bash
cd /path/to/llm-agent-lab
python scripts/cli_solve.py
```

常用参数：

1. `--api-url`：指定 `/solve` 地址（默认 `http://127.0.0.1:9000/solve`）  
2. `--model`：请求使用的模型名  
3. `--json`：直接输出完整 JSON  
4. `--verbose`：显示 Agent-1 和 Agent-2 的中间内容  

## API 说明

### 健康检查

```bash
curl --noproxy '*' -sS http://127.0.0.1:9000/health
```

### 求解接口

```bash
curl --noproxy '*' -sS http://127.0.0.1:9000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role":"user","content":"请简要解释 ReAct。"}],
    "gen": {"max_tokens": 512, "temperature": 0.7, "top_p": 0.8, "top_k": 20}
  }'
```

## 日志说明

每次调用 `POST /solve` 都会落一条 JSONL 日志到：

- `outputs/logs/run-YYYYMMDD.jsonl`

单条日志主要字段：

1. `trace_id`：请求链路 ID  
2. `latency_s`：整条流程耗时  
3. `agent_metrics`：三个 Agent 的时延与输入/输出长度  
4. `error`：异常信息（成功时为 `null`）

## 当前实现限制

1. 仅支持非流式返回（`stream=false`）。  
2. 图文能力依赖所加载模型本身：若模型/processor 不支持视觉输入，请求会自动降级为文本路径（保留 `<image>` 占位符）。  
3. 默认仍是最小可用实现，未覆盖 function calling、tool 调用等高级 OpenAI 字段。  

## 常见问题

1. `/solve` 返回错误：先确认 `8000` 的模型服务已启动。  
2. 本地请求超时：检查代理环境变量，`curl` 建议加 `--noproxy '*'`。  
3. GPU 资源冲突：通过 `CUDA_VISIBLE_DEVICES` 固定设备号。  
4. 端口占用：执行 `ss -ltnp | rg ':8000|:9000'` 排查。
