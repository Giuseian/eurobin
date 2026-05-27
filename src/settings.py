"""Central project configuration.

The settings loader reads the project-root `.env` file and exposes provider
configuration for Azure OpenAI and Gemini. Provider-specific credentials are
validated only when a model for that provider is used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)

GEMINI_ROBOTICS_ER_16_PREVIEW = "gemini-robotics-er-1.6-preview"

MODEL_PROVIDERS: dict[str, str] = {
    "o3": "azure",
    "gpt-5.2": "azure",
    GEMINI_ROBOTICS_ER_16_PREVIEW: "gemini",
}

MODEL_CAPABILITIES: dict[str, dict[str, bool]] = {
    "o3": {
        "supports_temperature": False,
        "supports_top_p": False,
    },
    "gpt-5.2": {
        "supports_temperature": True,
        "supports_top_p": True,
    },
    GEMINI_ROBOTICS_ER_16_PREVIEW: {
        "supports_temperature": False,
        "supports_top_p": False,
    },
}


@dataclass(frozen=True)
class Settings:
    endpoint: str
    api_key: str
    api_version: str
    deployments: dict[str, str]
    gemini_api_key: str
    project_root: Path


def load_settings() -> Settings:
    settings = Settings(
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "").strip(),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", "").strip(),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "").strip(),
        deployments={
            "o3": os.getenv("AZURE_OPENAI_DEPLOYMENT_O3", "").strip(),
            "gpt-5.2": os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT52", "").strip(),
        },
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        project_root=PROJECT_ROOT,
    )

    validate_settings(settings)
    return settings


def validate_settings(settings: Settings) -> None:
    missing: list[str] = []

    for model_name in MODEL_PROVIDERS:
        if model_name not in MODEL_CAPABILITIES:
            missing.append(f"MODEL_CAPABILITIES entry for model '{model_name}'")

    for model_name in MODEL_CAPABILITIES:
        if model_name not in MODEL_PROVIDERS:
            missing.append(f"MODEL_PROVIDERS entry for model '{model_name}'")

    if missing:
        raise ValueError(
            "Missing configuration values in .env or model metadata: " + ", ".join(missing)
        )


def validate_azure_settings(settings: Settings, model_name: str) -> None:
    missing: list[str] = []

    if not settings.endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not settings.api_key:
        missing.append("AZURE_OPENAI_API_KEY")
    if not settings.api_version:
        missing.append("AZURE_OPENAI_API_VERSION")

    if model_name not in settings.deployments:
        missing.append(f"AZURE_OPENAI_DEPLOYMENT for model '{model_name}'")
    elif not settings.deployments[model_name]:
        missing.append(f"AZURE_OPENAI_DEPLOYMENT for model '{model_name}'")

    if missing:
        raise ValueError(
            "Missing Azure OpenAI configuration values in .env: " + ", ".join(missing)
        )


def validate_gemini_settings(settings: Settings, model_name: str) -> None:
    if not settings.gemini_api_key:
        raise ValueError(
            f"Missing Gemini configuration for model '{model_name}': GEMINI_API_KEY"
        )


def validate_provider_settings(settings: Settings, model_name: str) -> None:
    provider = get_model_provider(model_name)

    if provider == "azure":
        validate_azure_settings(settings, model_name)
        return

    if provider == "gemini":
        validate_gemini_settings(settings, model_name)
        return

    raise ValueError(f"Unsupported provider '{provider}' for model '{model_name}'")


def resolve_deployment(settings: Settings, model_name: str) -> str:
    model_name = model_name.strip()

    if model_name not in settings.deployments:
        allowed = ", ".join(sorted(settings.deployments.keys()))
        raise ValueError(
            f"Unknown model '{model_name}'. Allowed values: {allowed}"
        )

    validate_azure_settings(settings, model_name)

    deployment_name = settings.deployments[model_name]
    if not deployment_name:
        raise ValueError(f"No deployment configured for model '{model_name}'")

    return deployment_name


def get_model_provider(model_name: str) -> str:
    normalized_model_name = model_name.strip()

    if normalized_model_name not in MODEL_PROVIDERS:
        allowed = ", ".join(sorted(MODEL_PROVIDERS.keys()))
        raise ValueError(
            f"Unknown model provider for '{normalized_model_name}'. "
            f"Allowed values: {allowed}"
        )

    return MODEL_PROVIDERS[normalized_model_name]


def get_model_capabilities(model_name: str) -> dict[str, bool]:
    normalized_model_name = model_name.strip()

    if normalized_model_name not in MODEL_CAPABILITIES:
        allowed = ", ".join(sorted(MODEL_CAPABILITIES.keys()))
        raise ValueError(
            f"Unknown model capabilities for '{normalized_model_name}'. "
            f"Allowed values: {allowed}"
        )

    return MODEL_CAPABILITIES[normalized_model_name]

