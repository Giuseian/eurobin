from __future__ import annotations

from pathlib import Path
from typing import Any

from azure_openai_client import call_azure_chat_completion
from gemini_client import call_gemini_completion
from settings import Settings, get_model_provider, validate_provider_settings


def call_llm_completion(
    settings: Settings,
    model_name: str,
    system_prompt: str,
    user_text: str,
    image_path: str | Path | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> dict[str, Any]:
    validate_provider_settings(settings, model_name)
    provider = get_model_provider(model_name)

    if provider == "azure":
        return call_azure_chat_completion(
            settings=settings,
            model_name=model_name,
            system_prompt=system_prompt,
            user_text=user_text,
            image_path=image_path,
            temperature=temperature,
            top_p=top_p,
        )

    if provider == "gemini":
        return call_gemini_completion(
            settings=settings,
            model_name=model_name,
            system_prompt=system_prompt,
            user_text=user_text,
            image_path=image_path,
            temperature=temperature,
            top_p=top_p,
        )

    raise ValueError(f"Unsupported provider '{provider}' for model '{model_name}'")
