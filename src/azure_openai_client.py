from __future__ import annotations

import base64
import time
from mimetypes import guess_type
from pathlib import Path
from typing import Any

from openai import AzureOpenAI

from settings import Settings, get_model_capabilities, resolve_deployment


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


def build_optional_model_params(
    model_name: str,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    if not 0.0 <= temperature <= 1.0:
        raise ValueError("temperature must be between 0.0 and 1.0")

    if not 0.0 <= top_p <= 1.0:
        raise ValueError("top_p must be between 0.0 and 1.0")

    capabilities = get_model_capabilities(model_name)
    optional_params: dict[str, Any] = {}

    if capabilities["supports_temperature"]:
        optional_params["temperature"] = temperature

    if capabilities["supports_top_p"]:
        optional_params["top_p"] = top_p

    return optional_params


def call_azure_chat_completion(
    settings: Settings,
    model_name: str,
    system_prompt: str,
    user_text: str,
    image_path: str | Path | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> dict[str, Any]:
    """
    Esegue una singola chiamata Azure OpenAI e restituisce:
    - model_name
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
            "content": build_user_content(
                user_text=user_text,
                image_path=image_path,
            ),
        },
    ]

    optional_params = build_optional_model_params(
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
    )

    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        **optional_params,
    )
    end_time = time.perf_counter()

    raw_content = response.choices[0].message.content

    return {
        "model_name": model_name,
        "deployment_name": deployment_name,
        "raw_response": raw_content,
        "execution_time_seconds": end_time - start_time,
    }
