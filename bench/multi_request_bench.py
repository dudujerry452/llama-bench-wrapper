#!/usr/bin/env python3
"""Measure llama-server under a deterministic multi-request workload."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import http.client
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "多场景" / "layout-transfer-v1" / "all.jsonl"


@dataclass(frozen=True)
class WorkItem:
    index: int
    row: dict[str, Any]


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic concurrent chat-completion requests against llama-server.",
        add_help=add_help,
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Chat completions endpoint, default: http://127.0.0.1:8080/v1/chat/completions",
    )
    parser.add_argument("--model-name", default="local", help="Model name sent in the request body")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="JSONL workload source")
    parser.add_argument(
        "--workload",
        choices=("mixed", "same-env", "prefix-local"),
        default="mixed",
        help="mixed: all environments; same-env: one environment; prefix-local: group by system prefix",
    )
    parser.add_argument("--env", default="env-00", help="Environment used by same-env or as a filter")
    parser.add_argument("--requests", type=int, default=100, help="Measured requests; 0 means all selected rows")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of closed-loop request workers")
    parser.add_argument(
        "--affinity",
        choices=("round-robin", "prefix"),
        default=None,
        help="Worker/slot assignment; prefix is selected automatically for prefix-local",
    )
    parser.add_argument("--seed", type=int, default=20261004, help="Random seed used for workload ordering")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum generated tokens per request")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Sampling top-p")
    parser.add_argument(
        "--cache-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request prompt KV reuse from llama-server",
    )
    parser.add_argument(
        "--pin-slots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send id_slot so each worker stays on a stable server slot",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup requests per worker, excluded from results")
    parser.add_argument("--think-time-ms", type=float, default=0.0, help="Delay after each measured request")
    parser.add_argument("--timeout-s", type=float, default=600.0, help="Per-request HTTP timeout")
    parser.add_argument("--health-timeout-s", type=float, default=60.0, help="Server readiness timeout")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for summary and JSONL results")
    parser.add_argument("--label", default="", help="Optional label stored in the result metadata")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory")
    return parser


def endpoint_url(value: str | None) -> str:
    return value or "http://127.0.0.1:8080/v1/chat/completions"


def origin_url(value: str) -> str:
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")


def health_url(value: str) -> str:
    return origin_url(value) + "/health"


def metrics_url(value: str) -> str:
    return origin_url(value) + "/metrics"


def slots_url(value: str) -> str:
    return origin_url(value) + "/slots"


def connection_for(url: str, timeout_s: float) -> tuple[http.client.HTTPConnection, str]:
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"unsupported URL: {url}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(parsed.hostname, port, timeout=timeout_s)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    return connection, path


def read_json_response(response: http.client.HTTPResponse) -> dict[str, Any]:
    body = response.read()
    if not body:
        return {}
    try:
        value = json.loads(body.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {"raw": body.decode("utf-8", errors="replace")[:4096]}
    return value if isinstance(value, dict) else {"value": value}


def get_text(url: str, timeout_s: float) -> tuple[int, str]:
    connection, path = connection_for(url, timeout_s)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, response.read().decode("utf-8", errors="replace")
    finally:
        connection.close()


def wait_for_health(url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = "server did not become ready"
    while time.monotonic() < deadline:
        try:
            status, body = get_text(url, min(5.0, max(0.1, deadline - time.monotonic())))
            if status == 200:
                return
            last_error = f"health status={status} body={body[:200]}"
        except OSError as error:
            last_error = str(error)
        time.sleep(0.25)
    raise TimeoutError(f"timed out waiting for {url}: {last_error}")


def environment_id(row: dict[str, Any]) -> str:
    row_id = str(row.get("id", ""))
    if row_id.startswith("ltv1-") and "-r" in row_id:
        return row_id[5:].split("-r", 1)[0]
    return "unknown"


def system_prefix(row: dict[str, Any]) -> str:
    messages = row.get("messages", [])
    for message in messages:
        if message.get("role") == "system":
            return str(message.get("content", ""))
    return ""


def system_fingerprint(row: dict[str, Any]) -> str:
    return hashlib.sha256(system_prefix(row).encode("utf-8")).hexdigest()[:16]


def load_rows(path: Path, env: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if env is None or environment_id(row) == env:
                rows.append(row)
    if not rows:
        filter_text = f" for {env}" if env else ""
        raise ValueError(f"no workload rows found in {path}{filter_text}")
    return rows


def ordered_rows(rows: list[dict[str, Any]], workload: str, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if workload == "prefix-local":
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[system_prefix(row)].append(row)
        keys = list(groups)
        rng.shuffle(keys)
        result: list[dict[str, Any]] = []
        for key in keys:
            group = list(groups[key])
            rng.shuffle(group)
            result.extend(group)
        return result
    result = list(rows)
    rng.shuffle(result)
    return result


def select_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    env_filter = args.env if args.workload == "same-env" else None
    if args.workload != "same-env" and args.env and args.env != "all":
        env_filter = args.env if getattr(args, "filter_env", False) else None
    rows = load_rows(args.data, env_filter)
    rows = ordered_rows(rows, args.workload, args.seed)
    if args.requests < 0:
        raise ValueError("--requests must be non-negative")
    if args.requests == 0:
        return rows
    selected: list[dict[str, Any]] = []
    cycle = 0
    while len(selected) < args.requests:
        cycle_rows = ordered_rows(rows, args.workload, args.seed + cycle + 1)
        selected.extend(cycle_rows)
        cycle += 1
    return selected[: args.requests]


def assign_work(items: list[WorkItem], worker_count: int, affinity: str) -> list[list[WorkItem]]:
    queues: list[list[WorkItem]] = [[] for _ in range(worker_count)]
    if affinity == "round-robin":
        for position, item in enumerate(items):
            queues[position % worker_count].append(item)
        return queues
    grouped: dict[str, list[WorkItem]] = defaultdict(list)
    for item in items:
        grouped[system_prefix(item.row)].append(item)
    for key, group in grouped.items():
        worker = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % worker_count
        queues[worker].extend(group)
    return queues


def request_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in row.get("messages", [])
        if message.get("role") in ("system", "user")
    ]


def request_payload(
    args: argparse.Namespace,
    row: dict[str, Any],
    slot: int | None,
    warmup: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model_name,
        "messages": request_messages(row),
        "max_tokens": 1 if warmup else args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_prompt": False if warmup else args.cache_prompt,
    }
    if args.pin_slots and slot is not None:
        payload["id_slot"] = slot
    return payload


def parse_stream_response(
    response: http.client.HTTPResponse,
    start_ns: int,
) -> tuple[dict[str, Any], int | None, int]:
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    first_data_ns: int | None = None
    first_token_ns: int | None = None
    event_count = 0
    for raw_line in response:
        now_ns = time.perf_counter_ns()
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data:"):
            continue
        value = line[5:].strip()
        if value == "[DONE]":
            continue
        try:
            event = json.loads(value)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_count += 1
        if first_data_ns is None:
            first_data_ns = now_ns
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        if isinstance(event.get("timings"), dict):
            timings = event["timings"]
        choices = event.get("choices") or []
        if choices and isinstance(choices[0], dict):
            choice = choices[0]
            delta = choice.get("delta") or {}
            content = delta.get("content") or delta.get("reasoning_content")
            tool_calls = delta.get("tool_calls")
            if content or tool_calls or choice.get("finish_reason"):
                if first_token_ns is None:
                    first_token_ns = now_ns
    response_data = {"usage": usage, "timings": timings, "event_count": event_count}
    first_ns = first_token_ns or first_data_ns
    return response_data, first_ns, time.perf_counter_ns()


def extract_metric(source: dict[str, Any], key: str) -> float | int | None:
    value = source.get(key)
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def request_once(
    endpoint: str,
    payload: dict[str, Any],
    timeout_s: float,
    item: WorkItem,
    worker: int,
    slot: int | None,
    warmup: bool,
) -> dict[str, Any]:
    start_ns = time.perf_counter_ns()
    status = 0
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    event_count = 0
    first_ns: int | None = None
    error = ""
    try:
        connection, path = connection_for(endpoint, timeout_s)
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        try:
            connection.request(
                "POST",
                path,
                body=body,
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            )
            response = connection.getresponse()
            status = response.status
            content_type = response.getheader("Content-Type", "")
            if status < 200 or status >= 300:
                response_body = response.read().decode("utf-8", errors="replace")
                error = response_body[:4096]
            elif "text/event-stream" in content_type:
                parsed, first_ns, _ = parse_stream_response(response, start_ns)
                usage = parsed.get("usage", {})
                timings = parsed.get("timings", {})
                event_count = int(parsed.get("event_count", 0))
            else:
                response_data = read_json_response(response)
                usage = response_data.get("usage") or {}
                timings = response_data.get("timings") or {}
                first_ns = time.perf_counter_ns()
        finally:
            connection.close()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    end_ns = time.perf_counter_ns()

    usage_details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else {}
    if not isinstance(usage_details, dict):
        usage_details = {}
    prompt_tokens = extract_metric(usage, "prompt_tokens")
    completion_tokens = extract_metric(usage, "completion_tokens")
    cached_tokens = extract_metric(usage_details, "cached_tokens")
    if cached_tokens is None:
        cached_tokens = extract_metric(timings, "cache_n")
    if prompt_tokens is None:
        cache_n = extract_metric(timings, "cache_n") or 0
        prompt_n = extract_metric(timings, "prompt_n") or 0
        prompt_tokens = cache_n + prompt_n
    if completion_tokens is None:
        completion_tokens = extract_metric(timings, "predicted_n")

    result = {
        "request_index": item.index,
        "dataset_id": item.row.get("id"),
        "environment": environment_id(item.row),
        "group_id": item.row.get("group_id"),
        "system_fingerprint": system_fingerprint(item.row),
        "worker": worker,
        "slot": slot,
        "warmup": warmup,
        "status": status,
        "ok": not error and 200 <= status < 300,
        "error": error,
        "latency_ms": (end_ns - start_ns) / 1_000_000,
        "ttft_ms": ((first_ns or end_ns) - start_ns) / 1_000_000 if status else None,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "prompt_ms": extract_metric(timings, "prompt_ms"),
        "generation_ms": extract_metric(timings, "predicted_ms"),
        "prompt_tokens_per_second": extract_metric(timings, "prompt_per_second"),
        "generation_tokens_per_second": extract_metric(timings, "predicted_per_second"),
        "event_count": event_count,
    }
    return result


async def run_worker(
    endpoint: str,
    args: argparse.Namespace,
    worker: int,
    queue: list[WorkItem],
) -> list[dict[str, Any]]:
    slot = worker if args.pin_slots else None
    if args.warmup > 0 and queue:
        for warmup_index in range(args.warmup):
            item = queue[warmup_index % len(queue)]
            payload = request_payload(args, item.row, slot, warmup=True)
            await asyncio.to_thread(request_once, endpoint, payload, args.timeout_s, item, worker, slot, True)
    results: list[dict[str, Any]] = []
    for item in queue:
        payload = request_payload(args, item.row, slot, warmup=False)
        result = await asyncio.to_thread(request_once, endpoint, payload, args.timeout_s, item, worker, slot, False)
        results.append(result)
        if args.think_time_ms > 0:
            await asyncio.sleep(args.think_time_ms / 1000.0)
    return results


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def numeric_values(results: list[dict[str, Any]], key: str) -> list[float]:
    return [float(result[key]) for result in results if isinstance(result.get(key), (int, float))]


def summarize(
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    measurement_s: float,
    selected_count: int,
    worker_count: int,
    output_dir: Path,
) -> dict[str, Any]:
    successful = [result for result in results if result.get("ok")]
    latency = numeric_values(successful, "latency_ms")
    ttft = numeric_values(successful, "ttft_ms")
    prompt_ms = numeric_values(successful, "prompt_ms")
    generation_ms = numeric_values(successful, "generation_ms")
    prompt_tokens = numeric_values(successful, "prompt_tokens")
    cached_tokens = numeric_values(successful, "cached_tokens")
    completion_tokens = numeric_values(successful, "completion_tokens")
    total_prompt_tokens = sum(prompt_tokens)
    total_cached_tokens = sum(cached_tokens)
    total_completion_tokens = sum(completion_tokens)
    request_count = len(results)
    success_count = len(successful)
    duration = max(measurement_s, 1e-9)

    def stats(values: list[float]) -> dict[str, float | None]:
        return {
            "count": len(values),
            "mean": statistics.fmean(values) if values else None,
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": max(values) if values else None,
        }

    affinity = args.affinity or ("prefix" if args.workload == "prefix-local" else "round-robin")
    return {
        "schema_version": 1,
        "label": args.label,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoint": args.url,
        "model_name": args.model_name,
        "dataset": str(args.data),
        "dataset_sha256": sha256_file(args.data),
        "workload": args.workload,
        "environment_filter": args.env if args.workload == "same-env" else None,
        "seed": args.seed,
        "selected_requests": selected_count,
        "measured_requests": request_count,
        "successful_requests": success_count,
        "failed_requests": request_count - success_count,
        "worker_count": worker_count,
        "concurrency": args.concurrency,
        "affinity": affinity,
        "pin_slots": args.pin_slots,
        "cache_prompt": args.cache_prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "warmup_per_worker": args.warmup,
        "think_time_ms": args.think_time_ms,
        "measurement_duration_s": measurement_s,
        "request_throughput_per_second": success_count / duration,
        "prompt_tokens_total": total_prompt_tokens,
        "cached_tokens_total": total_cached_tokens,
        "completion_tokens_total": total_completion_tokens,
        "prompt_tokens_per_second_wall": total_prompt_tokens / duration,
        "completion_tokens_per_second_wall": total_completion_tokens / duration,
        "total_tokens_per_second_wall": (total_prompt_tokens + total_completion_tokens) / duration,
        "cache_hit_requests": sum(1 for value in cached_tokens if value > 0),
        "cache_hit_ratio": (sum(1 for value in cached_tokens if value > 0) / len(cached_tokens)) if cached_tokens else None,
        "latency_ms": stats(latency),
        "ttft_ms": stats(ttft),
        "prompt_ms": stats(prompt_ms),
        "generation_ms": stats(generation_ms),
        "server_prompt_tokens_per_second": total_prompt_tokens / (sum(prompt_ms) / 1000.0) if prompt_ms and sum(prompt_ms) > 0 else None,
        "server_generation_tokens_per_second": total_completion_tokens / (sum(generation_ms) / 1000.0) if generation_ms and sum(generation_ms) > 0 else None,
        "result_files": {
            "requests": str(output_dir / "requests.jsonl"),
            "summary": str(output_dir / "summary.json"),
        },
    }


def prepare_output_dir(args: argparse.Namespace) -> Path:
    if getattr(args, "_output_dir_prepared", False):
        return args.output_dir.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = ROOT / "bench" / "results" / time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}; use --overwrite or choose another path")
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir
    args._output_dir_prepared = True
    return output_dir


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    args.url = endpoint_url(args.url)
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if not args.data.exists():
        raise FileNotFoundError(args.data)
    output_dir = prepare_output_dir(args)
    wait_for_health(health_url(args.url), args.health_timeout_s)

    selected = select_rows(args)
    worker_count = min(args.concurrency, len(selected))
    affinity = args.affinity or ("prefix" if args.workload == "prefix-local" else "round-robin")
    items = [WorkItem(index=index, row=row) for index, row in enumerate(selected)]
    queues = assign_work(items, worker_count, affinity)

    measurement_start = time.perf_counter()
    worker_results = await asyncio.gather(
        *(run_worker(args.url, args, worker, queue) for worker, queue in enumerate(queues))
    )
    measurement_s = time.perf_counter() - measurement_start
    results = [result for group in worker_results for result in group]
    results.sort(key=lambda result: int(result["request_index"]))
    summary = summarize(args, results, measurement_s, len(selected), worker_count, output_dir)
    summary["server_health_url"] = health_url(args.url)

    with (output_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def main() -> int:
    args = build_parser().parse_args()
    try:
        asyncio.run(run_benchmark(args))
    except (FileExistsError, FileNotFoundError, TimeoutError, ValueError, OSError) as error:
        print(f"benchmark error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
