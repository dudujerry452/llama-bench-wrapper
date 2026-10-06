#!/usr/bin/env python3
"""Start llama-server, run the multi-request benchmark, and collect diagnostics."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from multi_request_bench import (
    build_parser as build_benchmark_parser,
    get_text,
    health_url,
    metrics_url,
    prepare_output_dir,
    run_benchmark,
    slots_url,
    wait_for_health,
)


def build_parser() -> argparse.ArgumentParser:
    benchmark_parser = build_benchmark_parser(add_help=False)
    parser = argparse.ArgumentParser(
        description="Run a llama-server process and the multi-request benchmark.",
        parents=[benchmark_parser],
    )
    parser.add_argument("--model", type=Path, required=True, help="GGUF model path")
    parser.add_argument("--server", default="llama-server", help="llama-server executable")
    parser.add_argument("--host", default="127.0.0.1", help="Server bind address")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--slot-context", type=int, default=2048, help="Context budget per parallel slot")
    parser.add_argument("--ctx-size", type=int, default=0, help="Total server context; 0 means slot-context x concurrency")
    parser.add_argument("--batch-size", type=int, default=2048, help="Server logical batch size")
    parser.add_argument("--ubatch-size", type=int, default=512, help="Server physical batch size")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="Layers offloaded to GPU; use 0 for CPU")
    parser.add_argument("--threads", type=int, default=0, help="Server generation threads; 0 leaves default")
    parser.add_argument("--threads-batch", type=int, default=0, help="Server prompt threads; 0 leaves default")
    parser.add_argument(
        "--flash-attn",
        choices=("on", "off", "auto"),
        default=None,
        help="Override server Flash Attention mode",
    )
    parser.add_argument(
        "--cache-type-k",
        default=None,
        help="Override server KV cache K type",
    )
    parser.add_argument(
        "--cache-type-v",
        default=None,
        help="Override server KV cache V type",
    )
    parser.add_argument(
        "--kv-unified",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override unified KV mode",
    )
    parser.add_argument(
        "--cache-idle-slots",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override idle-slot prompt-cache behavior",
    )
    parser.add_argument("--cache-ram", type=int, default=None, help="Server prompt-cache RAM limit in MiB")
    parser.add_argument(
        "--cont-batching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable llama-server continuous batching",
    )
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra single server argument; repeat or use --server-arg=--flag",
    )
    parser.add_argument("--server-log", type=Path, default=None, help="Server log path")
    return parser


def server_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    ctx_size = args.ctx_size or args.slot_context * args.concurrency
    command = [
        args.server,
        "--model",
        str(args.model),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--ctx-size",
        str(ctx_size),
        "--parallel",
        str(args.concurrency),
        "--batch-size",
        str(args.batch_size),
        "--ubatch-size",
        str(args.ubatch_size),
        "--n-gpu-layers",
        str(args.n_gpu_layers),
        "--metrics",
        "--slots",
    ]
    command.append("--cont-batching" if args.cont_batching else "--no-cont-batching")
    command.append("--cache-prompt" if args.cache_prompt else "--no-cache-prompt")
    if args.threads > 0:
        command.extend(["--threads", str(args.threads)])
    if args.threads_batch > 0:
        command.extend(["--threads-batch", str(args.threads_batch)])
    if args.flash_attn:
        command.extend(["--flash-attn", args.flash_attn])
    if args.cache_type_k:
        command.extend(["--cache-type-k", args.cache_type_k])
    if args.cache_type_v:
        command.extend(["--cache-type-v", args.cache_type_v])
    if args.kv_unified is not None:
        command.append("--kv-unified" if args.kv_unified else "--no-kv-unified")
    if args.cache_idle_slots is not None:
        command.append("--cache-idle-slots" if args.cache_idle_slots else "--no-cache-idle-slots")
    if args.cache_ram is not None:
        command.extend(["--cache-ram", str(args.cache_ram)])
    log_path = args.server_log or output_dir / "server.log"
    command.extend(["--log-file", str(log_path)])
    command.extend(args.server_arg)
    return command


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


async def run(args: argparse.Namespace) -> int:
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    args.url = args.url or f"http://127.0.0.1:{args.port}/v1/chat/completions"
    output_dir = prepare_output_dir(args)
    command = server_command(args, output_dir)
    command_path = output_dir / "server-command.json"
    write_json(command_path, command)
    print(f"output: {output_dir}")
    print("server:", " ".join(command))

    log_handle = (output_dir / "server-process.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=Path.cwd(),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=os.environ.copy(),
    )
    summary: dict[str, object] | None = None
    try:
        wait_for_health(health_url(args.url), args.health_timeout_s)
        summary = await run_benchmark(args)
        for name, url in (("metrics.prom", metrics_url(args.url)), ("slots.json", slots_url(args.url))):
            try:
                status, body = get_text(url, args.timeout_s)
                (output_dir / name).write_text(body, encoding="utf-8")
                print(f"{url}: status={status}")
            except OSError as error:
                (output_dir / f"{name}.error").write_text(str(error) + "\n", encoding="utf-8")
        if summary is not None:
            summary["server"] = {
                "command": command,
                "model": str(args.model),
                "host": args.host,
                "port": args.port,
                "ctx_size": args.ctx_size or args.slot_context * args.concurrency,
                "slot_context": args.slot_context,
                "batch_size": args.batch_size,
                "ubatch_size": args.ubatch_size,
                "n_gpu_layers": args.n_gpu_layers,
                "cont_batching": args.cont_batching,
                "cache_prompt": args.cache_prompt,
                "cache_type_k": args.cache_type_k,
                "cache_type_v": args.cache_type_v,
                "kv_unified": args.kv_unified,
                "cache_idle_slots": args.cache_idle_slots,
                "cache_ram": args.cache_ram,
            }
            write_json(output_dir / "summary.json", summary)
        return 0
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
        log_handle.close()
        if process.returncode not in (0, 130, -signal.SIGINT):
            print(f"llama-server exited with code {process.returncode}", file=sys.stderr)


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(run(args))
    except (FileExistsError, FileNotFoundError, TimeoutError, ValueError, OSError) as error:
        print(f"server benchmark error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
