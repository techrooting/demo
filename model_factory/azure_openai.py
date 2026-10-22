import os
from typing import Any

from langchain.chat_models import init_chat_model


DEFAULT_AZURE_MODEL = "gpt-5.5-2026-04-24"
DEFAULT_AZURE_API_VERSION = "2024-03-01-preview"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_RETRIES = 3

#     model="openai:gpt-5.5",
def create_azure_openai_model(
    *,
    model: str = DEFAULT_AZURE_MODEL,
    api_version: str = DEFAULT_AZURE_API_VERSION,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Any:
    return init_chat_model(
        model=model,  # model 参数会作为 azure_deployment 使用
        model_provider="azure_openai",
        api_version=api_version,
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("API_KEY"),
        max_tokens=max_tokens,
        max_retries=max_retries,
    )
