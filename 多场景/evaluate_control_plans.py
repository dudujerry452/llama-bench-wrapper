#!/usr/bin/env python3
"""Rule-based validation and comparison for smart-home control plans."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


SCHEMA_DIR = Path(__file__).resolve().parents[1] / "scripts" / "data"
if str(SCHEMA_DIR) not in sys.path:
    sys.path.insert(0, str(SCHEMA_DIR))

from control_plan_schema import (  # noqa: E402
    compare_plans,
    parse_system_prompt,
    validate_plan,
)


def load_jsonl(path: Path) -> list[Any]:
    records: list[Any] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return records


def extract_messages(record: Any) -> tuple[str, str, str, str | None]:
    if not isinstance(record, dict):
        raise ValueError("record is not an object")

    identifier = record.get("id")
    if isinstance(record.get("messages"), list):
        messages = record["messages"]
        by_role = {
            item.get("role"): item.get("content", "")
            for item in messages
            if isinstance(item, dict)
        }
        if not all(isinstance(by_role.get(role), str) for role in ("system", "user", "assistant")):
            raise ValueError("messages must contain string system/user/assistant roles")
        return by_role["system"], by_role["user"], by_role["assistant"], identifier

    conversations = record.get("conversations")
    if isinstance(conversations, list):
        by_role = {
            item.get("from"): item.get("value", "")
            for item in conversations
            if isinstance(item, dict)
        }
        if not all(
            isinstance(by_role.get(role), str)
            for role in ("system", "user", "assistant")
        ):
            raise ValueError("conversations must contain string system/user/assistant roles")
        return by_role["system"], by_role["user"], by_role["assistant"], identifier

    raise ValueError("record has neither messages nor conversations")


def extract_prediction(record: Any) -> str:
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        raise ValueError("prediction record is not an object or string")

    for key in ("prediction", "output", "response", "text", "generated_text"):
        value = record.get(key)
        if isinstance(value, str):
            return value

    for key, role_key, content_key in (
        ("messages", "role", "content"),
        ("conversations", "from", "value"),
    ):
        messages = record.get(key)
        if isinstance(messages, list):
            for item in reversed(messages):
                if isinstance(item, dict) and item.get(role_key) == "assistant":
                    value = item.get(content_key)
                    if isinstance(value, str):
                        return value
    raise ValueError("prediction record has no prediction/output/assistant text")


def action_counter(parsed: Any) -> Counter:
    return Counter(
        (entry.device, entry.room, action.name, action.args)
        for entry in parsed.entries
        for action in entry.actions
    )


def evaluate(
    dataset_path: Path,
    prediction_path: Path | None,
    details_path: Path | None,
) -> dict[str, Any]:
    dataset = load_jsonl(dataset_path)
    predictions = load_jsonl(prediction_path) if prediction_path else None
    if predictions is not None and len(predictions) != len(dataset):
        raise ValueError(
            f"prediction rows ({len(predictions)}) do not match "
            f"dataset rows ({len(dataset)})"
        )

    counts = Counter()
    issue_counts: Counter[str] = Counter()
    comparison_counts: Counter[str] = Counter()
    details: list[dict[str, Any]] = []
    extra_action_rows = 0
    missing_action_rows = 0
    extra_action_total = 0
    missing_action_total = 0
    device_intersection_total = 0
    prediction_device_total = 0
    reference_device_total = 0
    device_room_intersection_total = 0
    prediction_device_room_total = 0
    reference_device_room_total = 0
    action_intersection_total = 0
    prediction_action_total = 0
    reference_action_total = 0

    for index, record in enumerate(dataset, 1):
        try:
            system, user, reference, identifier = extract_messages(record)
        except ValueError as exc:
            counts["dataset_parse_error_rows"] += 1
            details.append({"index": index, "dataset_error": str(exc)})
            continue

        prompt = parse_system_prompt(system)
        reference_validation = validate_plan(
            reference,
            prompt.services,
            prompt.relevant_devices,
        )
        if reference_validation.valid:
            counts["reference_valid_rows"] += 1
        else:
            counts["reference_invalid_rows"] += 1
            for issue in reference_validation.issues:
                issue_counts["reference:" + issue.split(":", 1)[0]] += 1

        prediction = reference if predictions is None else extract_prediction(predictions[index - 1])
        validation = validate_plan(
            prediction,
            prompt.services,
            prompt.relevant_devices,
        )
        counts["rows"] += 1
        if prediction.strip():
            counts["nonempty_prediction_rows"] += 1
        if not validation.parsed.issues and (
            validation.parsed.is_none or validation.parsed.entries
        ):
            counts["parse_valid_rows"] += 1
        if validation.valid:
            counts["valid_prediction_rows"] += 1
        for issue in validation.issues:
            issue_counts[issue.split(":", 1)[0]] += 1

        parsed_prediction = validation.parsed
        parsed_reference = reference_validation.parsed
        if parsed_prediction.is_none:
            counts["prediction_none_rows"] += 1
        if parsed_reference.is_none:
            counts["reference_none_rows"] += 1

        comparison = compare_plans(prediction, reference)
        for name, matched in comparison.items():
            if matched:
                comparison_counts[name] += 1

        pred_actions = action_counter(parsed_prediction)
        ref_actions = action_counter(parsed_reference)
        pred_devices = {entry.device for entry in parsed_prediction.entries}
        ref_devices = {entry.device for entry in parsed_reference.entries}
        pred_device_rooms = {
            (entry.device, entry.room) for entry in parsed_prediction.entries
        }
        ref_device_rooms = {
            (entry.device, entry.room) for entry in parsed_reference.entries
        }
        device_intersection_total += len(pred_devices & ref_devices)
        prediction_device_total += len(pred_devices)
        reference_device_total += len(ref_devices)
        device_room_intersection_total += len(pred_device_rooms & ref_device_rooms)
        prediction_device_room_total += len(pred_device_rooms)
        reference_device_room_total += len(ref_device_rooms)
        action_intersection_total += sum(
            (pred_actions & ref_actions).values()
        )
        prediction_action_total += sum(pred_actions.values())
        reference_action_total += sum(ref_actions.values())
        extra_actions = pred_actions - ref_actions
        missing_actions = ref_actions - pred_actions
        extra_count = sum(extra_actions.values())
        missing_count = sum(missing_actions.values())
        if extra_count:
            extra_action_rows += 1
            extra_action_total += extra_count
        if missing_count:
            missing_action_rows += 1
            missing_action_total += missing_count

        details.append(
            {
                "index": index,
                "id": identifier,
                "user": user,
                "reference": reference,
                "prediction": prediction,
                "prediction_valid": validation.valid,
                "prediction_issues": validation.issues,
                "reference_valid": reference_validation.valid,
                "reference_issues": reference_validation.issues,
                "comparison": comparison,
                "extra_action_count": extra_count,
                "missing_action_count": missing_count,
            }
        )

    denominator = counts["rows"] or 1
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path.resolve()),
        "predictions": str(prediction_path.resolve()) if prediction_path else None,
        "counts": dict(counts),
        "metrics": {
            "format_parse_rate": counts["parse_valid_rows"] / denominator,
            "executable_rate": counts["valid_prediction_rows"] / denominator,
            "nonempty_output_rate": counts["nonempty_prediction_rows"] / denominator,
            "reference_valid_rate": counts["reference_valid_rows"] / denominator,
            "none_accuracy": comparison_counts["none_match"] / denominator,
            "strict_exact_match": comparison_counts["strict_exact_match"] / denominator,
            "semantic_exact_match": comparison_counts["semantic_exact_match"] / denominator,
            "device_room_exact": comparison_counts["device_room_exact"] / denominator,
            "action_exact": comparison_counts["action_exact"] / denominator,
            "device_precision": (
                device_intersection_total / prediction_device_total
                if prediction_device_total
                else 1.0
            ),
            "device_recall": (
                device_intersection_total / reference_device_total
                if reference_device_total
                else 1.0
            ),
            "device_room_precision": (
                device_room_intersection_total / prediction_device_room_total
                if prediction_device_room_total
                else 1.0
            ),
            "device_room_recall": (
                device_room_intersection_total / reference_device_room_total
                if reference_device_room_total
                else 1.0
            ),
            "action_precision": (
                action_intersection_total / prediction_action_total
                if prediction_action_total
                else 1.0
            ),
            "action_recall": (
                action_intersection_total / reference_action_total
                if reference_action_total
                else 1.0
            ),
            "hallucinated_or_illegal_plan_rate": (
                (denominator - counts["valid_prediction_rows"]) / denominator
            ),
            "rows_with_extra_actions": extra_action_rows / denominator,
            "rows_with_missing_actions": missing_action_rows / denominator,
            "average_extra_actions": extra_action_total / denominator,
            "average_missing_actions": missing_action_total / denominator,
        },
        "issue_counts": dict(issue_counts.most_common()),
        "comparison_counts": dict(comparison_counts.most_common()),
    }

    if details_path:
        details_path.parent.mkdir(parents=True, exist_ok=True)
        with details_path.open("w", encoding="utf-8", newline="\n") as stream:
            for detail in details:
                stream.write(json.dumps(detail, ensure_ascii=False) + "\n")
        summary["details"] = str(details_path.resolve())
    return summary


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    counts = summary["counts"]
    metrics = summary["metrics"]
    lines = [
        "# Control-plan rule evaluation",
        "",
        f"- dataset: `{summary['dataset']}`",
        f"- predictions: `{summary['predictions'] or 'reference assistant labels'}`",
        f"- rows: `{counts.get('rows', 0)}`",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for name, value in metrics.items():
        lines.append(f"| `{name}` | {value:.4f} |")
    lines.extend(["", "## Issue counts", "", "| issue | rows |", "| --- | ---: |"])
    for issue, count in summary["issue_counts"].items():
        lines.append(f"| `{issue}` | {count} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--details", type=Path)
    args = parser.parse_args()

    summary = evaluate(args.dataset.resolve(), args.predictions.resolve() if args.predictions else None, args.details)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        markdown_path = args.output.with_suffix(".md")
        write_markdown(markdown_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
