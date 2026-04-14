"""命令行调用 `/solve` 的轻量客户端。

支持两种模式：
1) 单次提问：`--prompt "..."` 后直接输出结果
2) 交互模式：不传 `--prompt`，进入 REPL 循环
"""

import argparse
import json
import sys
from typing import Any, Dict

import requests


def call_solve(api_url: str, prompt: str, timeout: int, model: str) -> Dict[str, Any]:
    """调用编排服务并返回 JSON 结果。"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    session = requests.Session()
    # 禁用系统代理，避免本地地址请求被错误转发。
    session.trust_env = False
    resp = session.post(api_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def print_result(data: Dict[str, Any], verbose: bool = False) -> None:
    """格式化输出结果。"""
    if verbose:
        print("=== Agent-1 ===")
        print(data.get("agent1", ""))
        print("\n=== Agent-2 ===")
        print(data.get("agent2", ""))
        print("\n=== Final ===")
    print(data.get("final", ""))
    print(f"\n[trace_id] {data.get('trace_id', '-')}")
    if data.get("error"):
        print(f"[error] {data['error']}")


def run_once(args: argparse.Namespace) -> int:
    """单次提问模式。"""
    try:
        data = call_solve(args.api_url, args.prompt, args.timeout, args.model)
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            print_result(data, verbose=args.verbose)
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


def run_repl(args: argparse.Namespace) -> int:
    """交互式问答模式。输入 `exit` 或 `quit` 退出。"""
    print("LLM Agent Lab CLI 已启动，输入问题后回车；输入 exit/quit 退出。")
    while True:
        try:
            prompt = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n退出。")
            return 0
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            print("退出。")
            return 0

        try:
            data = call_solve(args.api_url, prompt, args.timeout, args.model)
            if args.json:
                print(json.dumps(data, ensure_ascii=False, indent=2))
            else:
                print_result(data, verbose=args.verbose)
        except Exception as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM Agent Lab /solve CLI")
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:9000/solve",
        help="编排服务地址，默认: http://127.0.0.1:9000/solve",
    )
    parser.add_argument(
        "--model",
        default="Qwen2-0.5B-Instruct",
        help="请求中的模型名（需与后端可用模型一致）",
    )
    parser.add_argument("--timeout", type=int, default=300, help="请求超时（秒）")
    parser.add_argument("--prompt", help="单次提问内容；不传则进入交互模式")
    parser.add_argument("--json", action="store_true", help="原样打印 JSON 结果")
    parser.add_argument("--verbose", action="store_true", help="展示 Agent-1/2 草稿和审稿内容")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.prompt:
        return run_once(args)
    return run_repl(args)


if __name__ == "__main__":
    raise SystemExit(main())
