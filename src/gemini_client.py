from __future__ import annotations

import time
from mimetypes import guess_type
from pathlib import Path
from typing import Any

from settings import Settings, get_model_capabilities, validate_gemini_settings


def read_image_part(image_path: str | Path) -> Any:
    try:
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "Missing Gemini SDK. Install it with: pip install google-genai"
        ) from exc

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    mime_type, _ = guess_type(str(image_path))
    if mime_type is None:
        mime_type = "application/octet-stream"

    return types.Part.from_bytes(
        data=image_path.read_bytes(),
        mime_type=mime_type,
    )


def build_gemini_config(
    model_name: str,
    system_prompt: str,
    temperature: float,
    top_p: float,
) -> Any:
    try:
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "Missing Gemini SDK. Install it with: pip install google-genai"
        ) from exc

    if not 0.0 <= temperature <= 1.0:
        raise ValueError("temperature must be between 0.0 and 1.0")

    if not 0.0 <= top_p <= 1.0:
        raise ValueError("top_p must be between 0.0 and 1.0")

    capabilities = get_model_capabilities(model_name)
    config: dict[str, Any] = {
        "system_instruction": system_prompt,
        "response_mime_type": "application/json",
    }

    if capabilities["supports_temperature"]:
        config["temperature"] = temperature

    if capabilities["supports_top_p"]:
        config["top_p"] = top_p

    return types.GenerateContentConfig(**config)


def make_gemini_client(settings: Settings) -> Any:
    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "Missing Gemini SDK. Install it with: pip install google-genai"
        ) from exc

    return genai.Client(api_key=settings.gemini_api_key)


def extract_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    raise ValueError("Gemini response did not contain text output.")


def call_gemini_completion(
    settings: Settings,
    model_name: str,
    system_prompt: str,
    user_text: str,
    image_path: str | Path | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> dict[str, Any]:
    validate_gemini_settings(settings, model_name)
    client = make_gemini_client(settings)

    contents: list[Any] = [user_text]
    if image_path is not None:
        contents.append(read_image_part(image_path))

    config = build_gemini_config(
        model_name=model_name,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
    )

    start_time = time.perf_counter()
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config,
    )
    end_time = time.perf_counter()

    return {
        "model_name": model_name,
        "deployment_name": model_name,
        "raw_response": extract_response_text(response),
        "execution_time_seconds": end_time - start_time,
    }
