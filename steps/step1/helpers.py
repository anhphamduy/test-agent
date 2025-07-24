import json
from typing import List, Dict

import streamlit as st
import os

from settings import get_openai_client


# ---------------------------------------------------------------------------
# Requirement extraction
# ---------------------------------------------------------------------------


def extract_requirements(text: str) -> List[Dict]:
    """Call OpenAI to extract a list of software requirements from raw text."""
    client = get_openai_client()

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
                                    "id": {
                                        "type": "integer",
                                        "description": "Unique requirement identifier, starting at 1.",
                                    },
                                    "name": {
                                        "type": "string",
                                        "description": "Short requirement description.",
                                    },
                                },
                                "required": ["id", "name"],
                            },
                        }
                    },
                    "required": ["requirements"],
                },
            },
        }
    ]

    messages = [
        {
            "role": "system",
            "content": "You are an expert business analyst extracting software requirements.",
        },
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

    if msg.tool_calls:
        for call in msg.tool_calls:
            if call.function.name == "extract_requirements":
                try:
                    args = json.loads(call.function.arguments)
                    return args.get("requirements", [])
                except json.JSONDecodeError:
                    st.warning("Failed to parse requirements from model response.")
    return []  # Fallback when parsing fails 