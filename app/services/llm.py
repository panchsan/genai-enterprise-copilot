from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

from app.config import settings
from app.services.logging_utils import get_logger
from app.services.resilience import retry_sync

logger = get_logger("app.llm")


def get_token_provider():
    return get_bearer_token_provider(
        AzureCliCredential(),
        "https://cognitiveservices.azure.com/.default",
    )


def get_azure_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_version=settings.AZURE_CHAT_API_VERSION,
        azure_ad_token_provider=get_token_provider(),
    )


def get_embeddings():
    from langchain_openai import AzureOpenAIEmbeddings

    logger.info("Initializing embeddings with Azure AD token")

    token_provider = get_bearer_token_provider(
        AzureCliCredential(),
        "https://cognitiveservices.azure.com/.default",
    )

    return AzureOpenAIEmbeddings(
        azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        api_version=settings.AZURE_EMBEDDING_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=token_provider,
    )


@retry_sync(max_attempts=3, delay_seconds=1.0)
def safe_chat_completion(client: AzureOpenAI, **kwargs):
    return client.chat.completions.create(**kwargs)