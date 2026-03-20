from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

from app.config import settings


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