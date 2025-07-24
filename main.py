import os
import json
from typing import List, Dict, Tuple

import streamlit as st
from openai import AzureOpenAI
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Set Streamlit page configuration early for wide layout
st.set_page_config(page_title="Requirement Extractor & Editor", page_icon="📄", layout="wide")

# --------------------------
# Helper functions
# --------------------------

def get_openai_client() -> AzureOpenAI:
    """Instantiate an Azure OpenAI client using env vars.

    Required environment variables:
    - AZURE_OPENAI_API_KEY
    - AZURE_OPENAI_ENDPOINT (e.g. https://your-resource.openai.azure.com/)
    - AZURE_OPENAI_API_VERSION (defaults to 2024-02-15-preview)
    Optional:
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


def extract_requirements(text: str) -> List[Dict]:
    """Call OpenAI to extract a list of software requirements from raw text."""
    client = get_openai_client()

    # Define the function schema for extraction
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_requirements",
                "description": "Return a list of software requirements, each with a unique id and concise name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "requirements": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "integer", "description": "Unique requirement identifier, starting at 1."},
                                    "name": {"type": "string", "description": "Short requirement description."}
                                },
                                "required": ["id", "name"]
                            }
                        }
                    },
                    "required": ["requirements"]
                },
            },
        }
    ]

    messages = [
        {"role": "system", "content": "You are an expert business analyst extracting software requirements."},
        {
            "role": "user",
            "content": (
                "From the document below, extract a de-duplicated list of distinct software requirements. "
                "Return them via the extract_requirements function only. Each requirement must be concise (≤ 15 words)."
            ),
        },
        {"role": "user", "content": text},
    ]

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-35-turbo"),
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    msg = response.choices[0].message

    # Parse the function call result
    if msg.tool_calls:
        for call in msg.tool_calls:
            if call.function.name == "extract_requirements":
                try:
                    args = json.loads(call.function.arguments)
                    return args.get("requirements", [])
                except json.JSONDecodeError:
                    st.warning("Failed to parse requirements from model response.")
    # Fallback: return empty list
    return []


def next_requirement_id(requirements: List[Dict]) -> int:
    """Compute the next available requirement ID."""
    return max((r["id"] for r in requirements), default=0) + 1


def apply_tool_call(requirements: List[Dict], call) -> Tuple[List[Dict], str]:
    """Apply the update_requirements function call, returning the new list and summary."""
    if call.function.name != "update_requirements":
        return requirements, ""

    try:
        args = json.loads(call.function.arguments)
        new_reqs = args.get("requirements", [])
        summary = args.get("summary", "")
        # Basic validation: ensure each item has id and name
        if all("id" in r and "name" in r for r in new_reqs):
            return new_reqs, summary
    except json.JSONDecodeError:
        pass
    return requirements, ""


def handle_chat(user_msg: str):
    """Process a user chat message, possibly modifying the requirements list via OpenAI function calls."""
    client = get_openai_client()
    requirements = st.session_state.requirements
    document_text = st.session_state.get("document_text", "")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "update_requirements",
                "description": "Replace the entire requirements list with a new list provided by the assistant.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "requirements": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "integer", "description": "Requirement identifier starting at 1."},
                                    "name": {"type": "string", "description": "Short requirement description."}
                                },
                                "required": ["id", "name"]
                            }
                        },
                        "summary": {
                            "type": "string",
                            "description": "Brief summary of changes made to the requirements based on user request."
                        }
                    },
                    "required": ["requirements", "summary"],
                },
            },
        }
    ]

    # Ensure no None content in stored history
    cleaned_history = [{**m, "content": m.get("content") or ""} for m in st.session_state.chat_history]

    # Compose conversation history for the model
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant helping the user refine the software requirements list. "
                "Whenever the user requests a change, respond ONLY by calling the update_requirements function with "
                "the complete new list of requirements AND a brief summary of what changes were made. If no change is needed, respond normally."
            ),
        },
        {
            "role": "system",
            "content": f"Current requirements list: {json.dumps(requirements)}",
        },
        {
            "role": "system",
            "content": (
                "Here is the full original document text the user uploaded. Use it as reference for context, "
                "but keep responses concise and structured via function calls when modifying requirements.\n\n" + document_text
            ),
        },
    ] + cleaned_history + [
        {"role": "user", "content": user_msg}
    ]

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-35-turbo"),
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    assistant_msg = response.choices[0].message
    assistant_content = assistant_msg.content or ""
    if assistant_content.strip():
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_content})

    # Apply any tool calls to the requirements list
    if assistant_msg.tool_calls:
        for call in assistant_msg.tool_calls:
            requirements, summary = apply_tool_call(requirements, call)
            if summary:
                st.session_state.chat_history.append({"role": "assistant", "content": f"**Summary of changes:** {summary}"})
        st.session_state.requirements = requirements


# --------------------------
# Streamlit UI
# --------------------------

def main():
    st.title("📄 Requirement Extractor & Editor")

    # NEW: delegate UI rendering to modular step files and return early to skip legacy code
    from helpers import init_session_state  # session-state utilities

    # Import the refactored step modules located under the 'steps' package
    from steps.step1 import view as step1
    from steps.step2 import view as step2
    from steps.step3 import view as step3

    init_session_state()

    if not st.session_state.requirements:
        step1.render()
    elif not st.session_state.get("test_cases"):
        step2.render()
    else:
        step3.render()
    return  # Prevent execution of legacy inline UI code below


if __name__ == "__main__":
    main()
