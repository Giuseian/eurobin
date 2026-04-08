from __future__ import annotations

import base64
import time
from mimetypes import guess_type
from pathlib import Path
from typing import Any

from openai import AzureOpenAI

from settings import Settings, resolve_deployment


def make_azure_client(settings: Settings) -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=settings.endpoint,
        api_key=settings.api_key,
        api_version=settings.api_version,
    )


def local_image_to_data_url(image_path: str | Path) -> str:
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    mime_type, _ = guess_type(str(image_path))
    if mime_type is None:
        mime_type = "application/octet-stream"

    with image_path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def build_user_content(
    user_text: str,
    image_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": user_text}
    ]

    if image_path is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": local_image_to_data_url(image_path),
                    "detail": "auto",
                },
            }
        )

    return content


def call_azure_chat_completion(
    settings: Settings,
    model_name: str,
    system_prompt: str,
    user_text: str,
    image_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Esegue una singola chiamata Azure OpenAI e restituisce:
    - deployment_name
    - raw_response
    - execution_time_seconds
    """
    deployment_name = resolve_deployment(settings, model_name)
    client = make_azure_client(settings)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": build_user_content(user_text=user_text, image_path=image_path),
        },
    ]

    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
    )
    end_time = time.perf_counter()

    raw_content = response.choices[0].message.content

    return {
        "model_name": model_name,
        "deployment_name": deployment_name,
        "raw_response": raw_content,
        "execution_time_seconds": end_time - start_time,
    }