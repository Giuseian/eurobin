""" `settings.py` is the central configuration file for the project.
It loads the `.env` file from the project root and reads the Azure OpenAI configuration from environment variables. In particular, it expects values such as the Azure endpoint, API key, API version, and the deployment names for the supported models.
It defines which models the project currently supports ```python o3 gpt-5.2``` and also stores what each model supports. For example, `o3` does not use `temperature` or `top_p`, while `gpt-5.2` does.
The main function is `load_settings()`. It builds a `Settings` object containing ```python endpoint api_key api_version deployments project_root```. Then it validates that all required values are present. If something is missing from `.env`, it raises an error before the pipeline starts.
It also provides helper functions like `resolve_deployment(...)`, which converts a logical model name such as `gpt-5.2` into the actual Azure deployment name, and `get_model_capabilities(...)`, which tells the rest of the code whether a model supports parameters like `temperature` and `top_p`.
In short: `settings.py` is where the project loads and validates Azure/OpenAI configuration, model deployment names, model capabilities, and the project root path."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)

MODEL_CAPABILITIES: dict[str, dict[str, bool]] = {
    "o3": {
        "supports_temperature": False,
        "supports_top_p": False,
    },
    "gpt-5.2": {
        "supports_temperature": True,
        "supports_top_p": True,
    },
}


@dataclass(frozen=True)
class Settings:
    endpoint: str
    api_key: str
    api_version: str
    deployments: dict[str, str]
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
        project_root=PROJECT_ROOT,
    )

    validate_settings(settings)
    return settings


def validate_settings(settings: Settings) -> None:
    missing: list[str] = []

    if not settings.endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not settings.api_key:
        missing.append("AZURE_OPENAI_API_KEY")
    if not settings.api_version:
        missing.append("AZURE_OPENAI_API_VERSION")

    for model_name, deployment_name in settings.deployments.items():
        if not deployment_name:
            missing.append(f"AZURE_OPENAI_DEPLOYMENT for model '{model_name}'")

        if model_name not in MODEL_CAPABILITIES:
            missing.append(f"MODEL_CAPABILITIES entry for model '{model_name}'")

    if missing:
        raise ValueError(
            "Missing configuration values in .env or model metadata: " + ", ".join(missing)
        )


def resolve_deployment(settings: Settings, model_name: str) -> str:
    model_name = model_name.strip()

    if model_name not in settings.deployments:
        allowed = ", ".join(sorted(settings.deployments.keys()))
        raise ValueError(
            f"Unknown model '{model_name}'. Allowed values: {allowed}"
        )

    deployment_name = settings.deployments[model_name]
    if not deployment_name:
        raise ValueError(f"No deployment configured for model '{model_name}'")

    return deployment_name


def get_model_capabilities(model_name: str) -> dict[str, bool]:
    normalized_model_name = model_name.strip()

    if normalized_model_name not in MODEL_CAPABILITIES:
        allowed = ", ".join(sorted(MODEL_CAPABILITIES.keys()))
        raise ValueError(
            f"Unknown model capabilities for '{normalized_model_name}'. "
            f"Allowed values: {allowed}"
        )

    return MODEL_CAPABILITIES[normalized_model_name]


