# from __future__ import annotations

# import os
# from dataclasses import dataclass
# from pathlib import Path
# from dotenv import load_dotenv


# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# ENV_PATH = PROJECT_ROOT / ".env"

# load_dotenv(dotenv_path=ENV_PATH)


# @dataclass(frozen=True)
# class Settings:
#     endpoint: str
#     api_key: str
#     api_version: str
#     default_model: str
#     default_temperature: float
#     deployments: dict[str, str]
#     project_root: Path
#     prompts_subdir: str
#     outputs_subdir: str


# def load_settings() -> Settings:
#     endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
#     api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
#     api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()

#     deployments = {
#         "o3": os.getenv("AZURE_OPENAI_DEPLOYMENT_O3", "").strip(),
#         "gpt-5.2": os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT52", "").strip(),
#     }

#     settings = Settings(
#         endpoint=endpoint,
#         api_key=api_key,
#         api_version=api_version,
#         default_model=os.getenv("DEFAULT_MODEL", "o3").strip(),
#         default_temperature=float(os.getenv("DEFAULT_TEMPERATURE", "0.0")),
#         deployments=deployments,
#         project_root=PROJECT_ROOT,
#         prompts_subdir=os.getenv("PROMPTS_SUBDIR", "prompts_shared").strip(),
#         outputs_subdir=os.getenv("OUTPUTS_SUBDIR", "outputs_shared").strip(),
#     )

#     validate_settings(settings)
#     return settings


# def validate_settings(settings: Settings) -> None:
#     missing = []

#     if not settings.endpoint:
#         missing.append("AZURE_OPENAI_ENDPOINT")
#     if not settings.api_key:
#         missing.append("AZURE_OPENAI_API_KEY")
#     if not settings.api_version:
#         missing.append("AZURE_OPENAI_API_VERSION")

#     for model_name, deployment in settings.deployments.items():
#         if not deployment:
#             missing.append(f"deployment for model '{model_name}'")

#     if missing:
#         raise ValueError(f"Missing configuration values: {', '.join(missing)}")


# def resolve_deployment(settings: Settings, model_name: str) -> str:
#     try:
#         deployment = settings.deployments[model_name]
#     except KeyError as exc:
#         allowed = ", ".join(settings.deployments.keys())
#         raise ValueError(
#             f"Unknown model '{model_name}'. Allowed values: {allowed}"
#         ) from exc

#     if not deployment:
#         raise ValueError(f"No deployment configured for model '{model_name}'")

#     return deployment


from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)


@dataclass(frozen=True)
class Settings:
    endpoint: str
    api_key: str
    api_version: str
    default_model: str
    default_temperature: float
    deployments: dict[str, str]
    project_root: Path
    outputs_root: str


def load_settings() -> Settings:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()

    deployments = {
        "o3": os.getenv("AZURE_OPENAI_DEPLOYMENT_O3", "").strip(),
        "gpt-5.2": os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT52", "").strip(),
    }

    settings = Settings(
        endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        default_model=os.getenv("DEFAULT_MODEL", "o3").strip(),
        default_temperature=float(os.getenv("DEFAULT_TEMPERATURE", "0.0")),
        deployments=deployments,
        project_root=PROJECT_ROOT,
        outputs_root=os.getenv("OUTPUTS_ROOT", "outputs").strip(),
    )

    validate_settings(settings)
    return settings


def validate_settings(settings: Settings) -> None:
    missing = []

    if not settings.endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not settings.api_key:
        missing.append("AZURE_OPENAI_API_KEY")
    if not settings.api_version:
        missing.append("AZURE_OPENAI_API_VERSION")

    for model_name, deployment in settings.deployments.items():
        if not deployment:
            missing.append(f"deployment for model '{model_name}'")

    if missing:
        raise ValueError(f"Missing configuration values: {', '.join(missing)}")


def resolve_deployment(settings: Settings, model_name: str) -> str:
    try:
        deployment = settings.deployments[model_name]
    except KeyError as exc:
        allowed = ", ".join(settings.deployments.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. Allowed values: {allowed}"
        ) from exc

    if not deployment:
        raise ValueError(f"No deployment configured for model '{model_name}'")

    return deployment