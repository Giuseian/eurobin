from __future__ import annotations

import base64
import time
from mimetypes import guess_type
from pathlib import Path
from typing import Any, Sequence

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

    if not image_path.is_file():
        raise ValueError(f"Image path is not a file: {image_path}")

    mime_type, _ = guess_type(str(image_path))
    if mime_type is None:
        mime_type = "application/octet-stream"

    with image_path.open("rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def normalize_image_paths(
    image_path: str | Path | None = None,
    image_paths: Sequence[str | Path] | None = None,
) -> list[str | Path]:
    """
    Normalizza gli input immagine mantenendo la retrocompatibilità.

    Casi supportati:
    - nessuna immagine;
    - una sola immagine tramite image_path;
    - una o più immagini tramite image_paths.

    Per evitare ambiguità, image_path e image_paths non possono essere
    valorizzati contemporaneamente.
    """
    if image_path is not None and image_paths is not None:
        raise ValueError(
            "Provide either image_path or image_paths, not both."
        )

    if image_paths is not None:
        normalized_paths = list(image_paths)

        if not normalized_paths:
            raise ValueError(
                "image_paths was provided but the sequence is empty."
            )

        return normalized_paths

    if image_path is not None:
        return [image_path]

    return []


def build_user_content(
    user_text: str,
    image_path: str | Path | None = None,
    image_paths: Sequence[str | Path] | None = None,
) -> list[dict[str, Any]]:
    """
    Costruisce il contenuto multimodale del messaggio utente.

    Retrocompatibilità:
        build_user_content(..., image_path="frame.png")

    Supporto multi-immagine:
        build_user_content(
            ...,
            image_paths=["i_pre.png", "i_post.png"],
        )

    L'ordine delle immagini nel messaggio coincide con l'ordine ricevuto
    in image_paths.
    """
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": user_text,
        }
    ]

    normalized_paths = normalize_image_paths(
        image_path=image_path,
        image_paths=image_paths,
    )

    for current_image_path in normalized_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": local_image_to_data_url(current_image_path),
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
    image_paths: Sequence[str | Path] | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> dict[str, Any]:
    """
    Esegue una singola chiamata Azure OpenAI.

    Supporta:
    - testo senza immagini;
    - una singola immagine tramite image_path;
    - una o più immagini tramite image_paths.

    image_path è mantenuto per retrocompatibilità con tutte le chiamate
    già esistenti nel progetto.

    Restituisce:
    - model_name
    - deployment_name
    - raw_response
    - execution_time_seconds
    """
    deployment_name = resolve_deployment(settings, model_name)
    client = make_azure_client(settings)

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": build_user_content(
                user_text=user_text,
                image_path=image_path,
                image_paths=image_paths,
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
