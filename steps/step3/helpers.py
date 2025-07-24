import json
import os
from typing import Dict, List, Tuple

import streamlit as st

from constants import DEFAULT_TEST_CASE_ITEM_SCHEMA
from settings import get_openai_client
from ..step2.helpers import generate_test_cases

# ---------------------------------------------------------------------------
# Schema inference helper
# ---------------------------------------------------------------------------


def infer_test_case_schema(user_msg: str) -> Tuple[Dict, List[int]]:
    """Determine schema changes and affected requirements via OpenAI function calling."""

    client = get_openai_client()

    current_schema = st.session_state.get(
        "test_case_schema", DEFAULT_TEST_CASE_ITEM_SCHEMA
    )
    current_test_cases = st.session_state.get("test_cases", [])
    requirements = st.session_state.get("requirements", [])

    tools = [
        {
            "type": "function",
            "function": {
                "name": "detect_schema_and_affected",
                "description": "Analyse the user's request and decide on schema changes and which requirements need test-case regeneration.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Short rationale",
                        },
                        "schema": {
                            "type": "string",
                            "description": "JSON string of the new item-level schema. Use '{}' if unchanged.",
                        },
                        "affected_requirement_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Requirement IDs whose test cases must be regenerated.",
                        },
                    },
                    "required": [
                        "schema",
                        "affected_requirement_ids",
                        "reason",
                    ],
                },
            },
        }
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior QA architect and JSON-Schema expert. Decide whether the item-level "
                "schema must change and which requirements need new test cases. Respond ONLY by calling "
                "the detect_schema_and_affected function. The `schema` argument must be a JSON string "
                "representing the full item-level schema, or '{}' if unchanged. Additionally, every property in the schema must have type 'string'."
            ),
        },
        {"role": "system", "content": f"Current schema: {json.dumps(current_schema)}"},
        {"role": "system", "content": f"Requirements list: {json.dumps(requirements)}"},
        {
            "role": "system",
            "content": f"Existing test cases: {json.dumps(current_test_cases)}",
        },
        {"role": "user", "content": user_msg},
    ]

    try:
        resp = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-35-turbo"),
            messages=messages,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "detect_schema_and_affected"},
            },
            temperature=0.0,
        )

        assistant_msg = resp.choices[0].message
        if not assistant_msg.tool_calls:
            return {}, []

        first_call = assistant_msg.tool_calls[0]
        try:
            args = json.loads(first_call.function.arguments)
            schema_str = args.get("schema", "{}")
            try:
                schema_obj = (
                    json.loads(schema_str) if isinstance(schema_str, str) else {}
                )
            except json.JSONDecodeError:
                schema_obj = {}
            affected_ids = args.get("affected_requirement_ids", [])

            if not isinstance(schema_obj, dict):
                schema_obj = {}
            if not isinstance(affected_ids, list):
                affected_ids = []

            affected_ids = [
                int(i)
                for i in affected_ids
                if isinstance(i, (int, float, str)) and str(i).isdigit()
            ]

            return schema_obj, affected_ids
        except Exception:
            return {}, []
    except Exception:
        return {}, []


# ---------------------------------------------------------------------------
# Test-case chat helpers
# ---------------------------------------------------------------------------


def apply_test_case_tool_call(test_cases: List[Dict], call) -> Tuple[List[Dict], str]:
    """Handle the update_test_cases function call coming from the model."""

    if call.function.name != "update_test_cases":
        return test_cases, ""

    try:
        args = json.loads(call.function.arguments)
        new_cases = args.get("test_cases", None)
        summary = args.get("summary", "")

        if new_cases is None:
            return test_cases, ""

        if (
            isinstance(new_cases, list)
            and len(new_cases) > 0
            and all(isinstance(c, dict) for c in new_cases)
        ):
            return new_cases, summary
        return test_cases, ""
    except json.JSONDecodeError:
        pass
    return test_cases, ""


def handle_test_case_chat(user_msg: str):
    """Process a chat message related to test cases, allowing the model to modify them via function calls."""

    client = get_openai_client()

    test_cases = st.session_state.test_cases

    # 1) Schema inference
    current_schema: Dict = st.session_state.get(
        "test_case_schema", DEFAULT_TEST_CASE_ITEM_SCHEMA
    )
    inferred_schema, affected_ids = infer_test_case_schema(user_msg)
    if inferred_schema:
        current_schema = inferred_schema
        st.session_state.test_case_schema = current_schema

    # 2) Regenerate test cases for affected requirements if needed
    if affected_ids:
        requirements: List[Dict] = st.session_state.get("requirements", [])
        affected_requirements = [r for r in requirements if r["id"] in affected_ids]
        if affected_requirements:
            # Pass the current user query so the test-case generation has full context
            new_cases = generate_test_cases(affected_requirements, current_schema, user_msg)

            # Replace cases for these requirements in the global list
            test_cases = [
                c for c in test_cases if c.get("requirement_id") not in affected_ids
            ]
            test_cases.extend(new_cases)

            st.session_state.test_cases = test_cases

            st.session_state.test_case_chat_history.append(
                {
                    "role": "assistant",
                    "content": (
                        "Regenerated test cases for requirements "
                        + ", ".join(map(str, affected_ids))
                        + "."
                    ),
                }
            )
