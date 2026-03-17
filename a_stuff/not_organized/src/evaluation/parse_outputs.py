from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "outputs_official"
RAW_PARSED_ROOT = PROJECT_ROOT / "evaluation_outputs" / "raw_parsed"


def infer_input_mode(image_path: str | None) -> str:
    if image_path is None or str(image_path).strip() == "":
        return "structured_text"
    return "image"


def safe_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def collect_output_files(outputs_root: Path) -> list[Path]:
    return sorted(
        [
            p for p in outputs_root.rglob("*.json")
            if p.is_file()
        ]
    )


def parse_single_file(file_path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "source_file": str(file_path.relative_to(PROJECT_ROOT)),
            "scenario_id": None,
            "task_text": None,
            "model_name": None,
            "prompt_filename": None,
            "repeat_idx": None,
            "image_path": None,
            "input_mode": None,
            "json_parse_ok": False,
            "raw_response": None,
            "parsed_json": None,
            "timestamp_utc": None,
            "inference_time_sec": None,
            "is_error_file": True,
            "file_read_error": str(exc),
        }

    image_path = payload.get("image_path")
    input_mode = infer_input_mode(image_path)

    is_error_file = "error" in payload

    row = {
        "source_file": str(file_path.relative_to(PROJECT_ROOT)),
        "scenario_id": payload.get("scenario_id"),
        "task_text": payload.get("task_text"),
        "model_name": payload.get("model_name"),
        "prompt_filename": payload.get("prompt_filename"),
        "repeat_idx": payload.get("repeat_idx"),
        "image_path": image_path,
        "input_mode": input_mode,
        "json_parse_ok": payload.get("json_parse_ok"),
        "raw_response": payload.get("raw_response"),
        "parsed_json": payload.get("parsed_json"),
        "timestamp_utc": payload.get("timestamp_utc"),
        "inference_time_sec": payload.get("inference_time_sec"),
        "is_error_file": is_error_file,
        "file_read_error": None,
    }

    return row


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_file",
        "scenario_id",
        "task_text",
        "model_name",
        "prompt_filename",
        "repeat_idx",
        "image_path",
        "input_mode",
        "json_parse_ok",
        "raw_response",
        "parsed_json",
        "timestamp_utc",
        "inference_time_sec",
        "is_error_file",
        "file_read_error",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            row_to_write = row.copy()
            row_to_write["parsed_json"] = safe_json_dumps(row_to_write["parsed_json"])
            row_to_write["raw_response"] = (
                row_to_write["raw_response"] if row_to_write["raw_response"] is not None else ""
            )
            writer.writerow(row_to_write)


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    if not OUTPUTS_ROOT.exists():
        raise FileNotFoundError(f"Outputs folder not found: {OUTPUTS_ROOT}")

    files = collect_output_files(OUTPUTS_ROOT)

    if not files:
        print(f"No JSON files found under: {OUTPUTS_ROOT}")
        return

    rows: list[dict[str, Any]] = []

    for file_path in files:
        row = parse_single_file(file_path)
        if row is not None:
            rows.append(row)

    csv_path = RAW_PARSED_ROOT / "per_run_raw.csv"
    jsonl_path = RAW_PARSED_ROOT / "per_run_raw.jsonl"

    write_csv(rows, csv_path)
    write_jsonl(rows, jsonl_path)

    print(f"Parsed files: {len(files)}")
    print(f"Rows written: {len(rows)}")
    print(f"CSV saved to: {csv_path}")
    print(f"JSONL saved to: {jsonl_path}")


if __name__ == "__main__":
    main()