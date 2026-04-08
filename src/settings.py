from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)


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

    if missing:
        raise ValueError(
            "Missing configuration values in .env: " + ", ".join(missing)
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