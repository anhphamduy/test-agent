import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import streamlit as st

from constants import DEFAULT_TEST_CASE_ITEM_SCHEMA
from settings import get_openai_client

# ---------------------------------------------------------------------------
# Utilities shared within Step-2
# ---------------------------------------------------------------------------


def next_requirement_id(requirements: List[Dict]) -> int:
    """Return the next available requirement ID."""
    return max((r["id"] for r in requirements), default=0) + 1


# ---------------------------------------------------------------------------
# Chat-based requirement editing helpers
# ---------------------------------------------------------------------------

def apply_tool_call(requirements: List[Dict], call) -> Tuple[List[Dict], str]:
    """Apply an `update_requirements` tool call coming from the LLM."""

    if call.function.name != "update_requirements":
        return requirements, ""

    try:
        args = json.loads(call.function.arguments)
        new_reqs = args.get("requirements", [])
        summary = args.get("summary", "")
        if all("id" in r and "name" in r for r in new_reqs):
            return new_reqs, summary
    except json.JSONDecodeError:
        pass
    return requirements, ""


def handle_chat(user_msg: str):
    """Process a user chat message, possibly modifying the requirements via OpenAI function calls."""

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
                                    "id": {
                                        "type": "integer",
                                        "description": "Requirement identifier starting at 1.",
                                    },
                                    "name": {
                                        "type": "string",
                                        "description": "Short requirement description.",
                                    },
                                },
                                "required": ["id", "name"],
                            },
                        },
                        "summary": {
                            "type": "string",
                            "description": "Brief summary of changes made to the requirements based on user request.",
                        },
                    },
                    "required": ["requirements", "summary"],
                },
            },
        }
    ]

    cleaned_history = [
        {**m, "content": m.get("content") or ""} for m in st.session_state.chat_history
    ]

    messages = (
        [
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
                    "but keep responses concise and structured via function calls when modifying requirements.\n\n"
                    + document_text
                ),
            },
        ]
        + cleaned_history
        + [{"role": "user", "content": user_msg}]
    )

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

    if assistant_msg.tool_calls:
        for call in assistant_msg.tool_calls:
            requirements, summary = apply_tool_call(requirements, call)
            if summary:
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": f"**Summary of changes:** {summary}",
                    }
                )
        st.session_state.requirements = requirements


# ---------------------------------------------------------------------------
# Test-case generation helpers
# ---------------------------------------------------------------------------


def _generate_test_cases_for_requirement(requirement: Dict, item_schema: Dict) -> List[Dict]:
    """Internal helper to call OpenAI and generate test cases for a single requirement."""

    client = get_openai_client()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "generate_test_cases",
                "description": (
                    "Return 3-5 manual test cases for the given requirement, strictly following the provided JSON schema."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "test_cases": {
                            "type": "array",
                            "items": item_schema,
                        }
                    },
                    "required": ["test_cases"],
                },
            },
        }
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior QA engineer tasked with designing clear, executable manual test cases. "
                "The test cases **must** conform to the following JSON schema for each object: "
                f"```json\n{json.dumps(item_schema)}\n```"
            ),
        },
        {
            "role": "user",
            "content": (
                "Generate 3-5 test cases for the requirement below and return them **only** via the "
                "generate_test_cases function.\n\n"
                + f"Requirement: {requirement['name']}"
            ),
        },
    ]

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-35-turbo"),
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    msg = response.choices[0].message

    cases: List[Dict] = []
    if msg.tool_calls:
        for call in msg.tool_calls:
            if call.function.name == "generate_test_cases":
                try:
                    args = json.loads(call.function.arguments)
                    raw_cases = args.get("test_cases", [])

                    # Post-process: ensure each case includes requirement_id / requirement_name
                    for c in raw_cases:
                        c["requirement_id"] = requirement["id"]
                        c["requirement_name"] = requirement["name"]
                        cases.append(c)
                except json.JSONDecodeError:
                    pass  # Skip badly formatted responses

    return cases


def generate_test_cases(requirements: List[Dict], item_schema: Dict) -> List[Dict]:
    """Generate test cases for each requirement using parallel OpenAI calls."""

    if not requirements:
        return []

    results: List[Dict] = []

    with ThreadPoolExecutor(max_workers=min(20, len(requirements))) as executor:
        future_to_req = {
            executor.submit(_generate_test_cases_for_requirement, r, item_schema): r
            for r in requirements
        }

        for future in as_completed(future_to_req):
            try:
                results.extend(future.result())
            except Exception:
                # Ignore failures for individual requirements
                continue

    return results 