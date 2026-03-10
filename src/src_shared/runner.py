from __future__ import annotations

import base64
import json
import time 
from datetime import datetime, timezone
from mimetypes import guess_type
from typing import Any

from .azure_client import make_azure_client
from .config import Settings, resolve_deployment
from .prompts import build_user_block, load_system_prompt


def local_image_to_data_url(image_path: str) -> str:
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def try_parse_json(text: str) -> tuple[bool, Any]:
    try:
        return True, json.loads(text)
    except Exception:
        return False, None


def run_single_experiment(
    settings: Settings,
    model_name: str,
    prompt_filename: str,
    image_path: str,
    task_text: str,
    temperature: float | None,
) -> dict[str, Any]:
    deployment_name = resolve_deployment(settings, model_name)
    client = make_azure_client(settings)

    system_prompt = load_system_prompt(settings, prompt_filename)
    user_block = build_user_block(task_text)
    data_url = local_image_to_data_url(image_path)

    request_kwargs = {
        "model": deployment_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_block},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "auto"},
                    },
                ],
            },
        ],
    }

    # Reasoning models like o3 do not support custom temperature values.
    if model_name not in {"o3", "gpt-5.2", "gpt-5.1"} and temperature is not None:
        request_kwargs["temperature"] = temperature

    start_time = time.perf_counter()

    response = client.chat.completions.create(**request_kwargs)

    end_time = time.perf_counter()

    inference_time_sec = end_time - start_time

    raw_content = response.choices[0].message.content
    parsed_ok, parsed_json = try_parse_json(raw_content)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "deployment_name": deployment_name,
        "prompt_filename": prompt_filename,
        "image_path": image_path,
        "task_text": task_text,
        "temperature": temperature,
        "inference_time_sec": inference_time_sec,
        "raw_response": raw_content,
        "json_parse_ok": parsed_ok,
        "parsed_json": parsed_json,
    }