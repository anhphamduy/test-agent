import os

import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

# Automatically load values from a .env file when present so that local
# development "just works" without exporting variables manually.
load_dotenv()


# ---------------------------------------------------------------------------
# Azure OpenAI helper
# ---------------------------------------------------------------------------


def get_openai_client() -> AzureOpenAI:
    """Instantiate an Azure OpenAI client using environment variables.

    Required environment variables:
        - AZURE_OPENAI_API_KEY
        - AZURE_OPENAI_ENDPOINT (e.g. https://your-resource.openai.azure.com/)

    Optional:
        - AZURE_OPENAI_API_VERSION (defaults to 2024-12-01-preview)
        - AZURE_OPENAI_CHAT_DEPLOYMENT (deployment name for chat/completions; defaults to 'gpt-35-turbo')
    """

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    if not api_key or not endpoint:
        st.error(
            "Missing Azure OpenAI configuration. Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT."
        )
        st.stop()

    return AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version) 