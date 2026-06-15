from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_MODEL = "text-embedding-3-large"


def load_environment() -> None:
    load_dotenv(dotenv_path=ENV_PATH)


def make_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in environment or .env")

    return OpenAI(api_key=api_key)


def make_azure_openai_client() -> AzureOpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()

    missing = []
    if not endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not api_key:
        missing.append("AZURE_OPENAI_API_KEY")
    if not api_version:
        missing.append("AZURE_OPENAI_API_VERSION")

    if missing:
        raise ValueError("Missing Azure OpenAI configuration: " + ", ".join(missing))

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def resolve_azure_embedding_deployment(model: str) -> str:
    deployment = (
        os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "").strip()
        or os.getenv("AZURE_OPENAI_DEPLOYMENT_TEXT_EMBEDDING_3_LARGE", "").strip()
    )

    if not deployment:
        raise ValueError(
            "Missing Azure embedding deployment. Add one of these variables to .env: "
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "
            "AZURE_OPENAI_DEPLOYMENT_TEXT_EMBEDDING_3_LARGE. "
            f"The deployment should point to the model '{model}'."
        )

    return deployment


def create_embedding(
    text: str,
    provider: str,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    text = text.replace("\n", " ").strip()
    if not text:
        raise ValueError("Input text cannot be empty")

    if provider == "azure":
        client = make_azure_openai_client()
        deployment_name = resolve_azure_embedding_deployment(model)
        response = client.embeddings.create(
            model=deployment_name,
            input=text,
        )
        used_model = deployment_name
    elif provider == "openai":
        client = make_openai_client()
        response = client.embeddings.create(
            model=model,
            input=text,
        )
        used_model = model
    else:
        raise ValueError("provider must be 'azure' or 'openai'")

    embedding = response.data[0].embedding

    return {
        "provider": provider,
        "model": model,
        "api_model_or_deployment": used_model,
        "input_text": text,
        "embedding_dimensions": len(embedding),
        "embedding": embedding,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a text embedding with text-embedding-3-large."
    )
    parser.add_argument(
        "text",
        nargs="?",
        default="Questo e' un semplice esempio di embedding.",
        help="Text to convert into an embedding.",
    )
    parser.add_argument(
        "--provider",
        choices=["azure", "openai"],
        default="azure",
        help="Use Azure OpenAI or the OpenAI API.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Embedding model name. For Azure, this is used only for metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path where the full embedding JSON will be written.",
    )
    return parser.parse_args()


def main() -> None:
    load_environment()
    args = parse_args()

    result = create_embedding(
        text=args.text,
        provider=args.provider,
        model=args.model,
    )

    print(f"Provider: {result['provider']}")
    print(f"Model: {result['model']}")
    print(f"API model/deployment: {result['api_model_or_deployment']}")
    print(f"Embedding dimensions: {result['embedding_dimensions']}")
    print(f"First 10 values: {result['embedding'][:10]}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Full embedding written to: {args.output}")


if __name__ == "__main__":
    main()
