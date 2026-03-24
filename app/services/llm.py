import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

from app.config import settings
from app.services.logging_utils import get_logger
from app.services.resilience import retry_sync

logger = get_logger("app.llm")


def _log_auth_env():
    logger.info(
        "Auth env check | "
        f"AZURE_CLIENT_ID={'set' if os.getenv('AZURE_CLIENT_ID') else 'missing'} | "
        f"AZURE_TENANT_ID={'set' if os.getenv('AZURE_TENANT_ID') else 'missing'} | "
        f"AZURE_CLIENT_SECRET={'set' if os.getenv('AZURE_CLIENT_SECRET') else 'missing'} | "
        f"AZURE_OPENAI_ENDPOINT={'set' if os.getenv('AZURE_OPENAI_ENDPOINT') else 'missing'}"
    )


def get_token_provider():
    _log_auth_env()

    credential = DefaultAzureCredential(
        exclude_interactive_browser_credential=True,
    )

    return get_bearer_token_provider(
        credential,
        "https://cognitiveservices.azure.com/.default",
    )


def get_azure_openai_client() -> AzureOpenAI:
    logger.info("Initializing Azure OpenAI client with Microsoft Entra ID")
    return AzureOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_version=settings.AZURE_CHAT_API_VERSION,
        azure_ad_token_provider=get_token_provider(),
    )


def get_embeddings():
    from langchain_openai import AzureOpenAIEmbeddings

    logger.info("Initializing embeddings with Microsoft Entra ID")

    return AzureOpenAIEmbeddings(
        azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        api_version=settings.AZURE_EMBEDDING_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=get_token_provider(),
    )


@retry_sync(max_attempts=3, delay_seconds=1.0)
def safe_chat_completion(client: AzureOpenAI, **kwargs):
    return client.chat.completions.create(**kwargs)