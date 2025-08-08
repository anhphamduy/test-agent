import json
import os
from typing import Dict, List, Tuple, Any

import streamlit as st

from constants import DEFAULT_TEST_CASE_ITEM_SCHEMA
from settings import get_openai_client
from ..step2.helpers import generate_test_cases

# ---------------------------------------------------------------------------
# Schema inference helper
# ---------------------------------------------------------------------------


def infer_test_case_schema(user_msg: str) -> Tuple[Dict, List[int], str, str]:
    """Determine schema changes and affected requirements via OpenAI function calling."""

    client = get_openai_client()

    current_schema = st.session_state.get(
        "test_case_schema", DEFAULT_TEST_CASE_ITEM_SCHEMA
    )
    current_test_cases = st.session_state.get("test_cases", [])
    requirements = st.session_state.get("requirements", [])

    # Ensure every property is required in the schema passed to the model
    full_item_schema = json.loads(json.dumps(current_schema))  # deep copy
    full_item_schema["required"] = list(full_item_schema.get("properties", {}).keys())

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
                            "description": "JSON string of the new item-level schema, include any instructions in the field description. Use '{}' if unchanged.",
                        },
                        "sample_test_case": {
                            "type": "string",
                            "description": "JSON string of ONE sample test case object that follows the schema. Use '{}' if unchanged.",
                        },
                        "affected_requirement_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Requirement IDs whose test cases must be regenerated.",
                        },
                        "instruction": {
                            "type": "string",
                            "description": "Concise instruction that should be forwarded to the test-case generation step instead of the full chat history.",
                        },
                    },
                    "required": [
                        "schema",
                        "affected_requirement_ids",
                        "sample_test_case",
                        "reason",
                        "instruction",
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
                "the detect_schema_and_affected function **ONLY when** the user is requesting changes that affect the test-case schema or require regenerating test cases. "
                "If the user's message is purely informational (Q&A) and no adjustments are needed, respond normally **without** calling the function and end your reply with a question asking if the user wants these ideas applied to the test-case table (e.g. 'Would you like me to update the test cases accordingly?'). "
                "When you do call the function, it **must** include an `instruction` field that contains a concise directive for how to generate the test cases (e.g. 'Add edge cases'). "
                "The `schema` argument must be a JSON string representing the full item-level schema, or '{}' if unchanged. Additionally, every property in the schema must have type 'string'. "
                "Important: If the user asks to delete test cases (or a requirement) that does not exist, do NOT call any function. Instead, reply normally with an apology (e.g. 'Sorry, there are no test cases for requirement X.') so the user is informed."
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
            tool_choice="auto",
            temperature=0.0,
        )
        assistant_msg = resp.choices[0].message
        if not assistant_msg.tool_calls:
            # Pure chat response; return its content so caller can display it
            return {}, [], "", assistant_msg.content or ""

        first_call = assistant_msg.tool_calls[0]
        try:
            args = json.loads(first_call.function.arguments)
            schema_str = args.get("schema", "{}")
            sample_str = args.get("sample_test_case", "{}")
            try:
                schema_obj = (
                    json.loads(schema_str) if isinstance(schema_str, str) else {}
                )
            except json.JSONDecodeError:
                schema_obj = {}
            try:
                sample_obj = (
                    json.loads(sample_str) if isinstance(sample_str, str) else {}
                )
            except json.JSONDecodeError:
                sample_obj = {}
            affected_ids = args.get("affected_requirement_ids", [])
            instruction = args.get("instruction", "")

            if not isinstance(schema_obj, dict):
                schema_obj = {}
            if not isinstance(sample_obj, dict):
                sample_obj = {}
            if not isinstance(affected_ids, list):
                affected_ids = []

            affected_ids = [
                int(i)
                for i in affected_ids
                if isinstance(i, (int, float, str)) and str(i).isdigit()
            ]

            # Store sample in session for later consistency
            if sample_obj:
                st.session_state["test_case_sample"] = sample_obj
            else:
                if schema_obj:
                    st.session_state["test_case_sample"] = _build_generic_sample(schema_obj)

            return schema_obj, affected_ids, instruction, ""
        except Exception:
            return {}, [], "", ""
    except Exception:
        return {}, [], "", ""


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


def handle_test_case_chat(latest_user_msg: str):
    """Process a chat message related to test cases. This now:

    1. Aggregates previous user messages so the LLM receives full context.
    2. Forces the assistant to respond via the `update_test_cases` function when modifications are needed.
    3. Still supports schema inference & selective regeneration as before.
    """

    client = get_openai_client()

    # -----------------------------------------------------------
    # Build an aggregated conversation string containing BOTH past
    # user *and* assistant messages in chronological order followed
    # by the latest user message. This richer context helps the LLM
    # understand the full dialogue, not just user utterances.
    # -----------------------------------------------------------
    conversation_lines = [
        f"{msg.get('role', '').capitalize()}: {msg.get('content', '')}"
        for msg in st.session_state.test_case_chat_history
    ]
    # Append the current user message as the last line
    conversation_lines.append(f"User: {latest_user_msg}")
    aggregated_user_msg = "\n".join(conversation_lines)

    test_cases = st.session_state.test_cases

    # --------------------------------------------------------------------
    # 1) Schema inference (may trigger regeneration of some test cases)
    # --------------------------------------------------------------------
    current_schema: Dict = st.session_state.get(
        "test_case_schema", DEFAULT_TEST_CASE_ITEM_SCHEMA
    )
    inferred_schema, affected_ids, instruction, assistant_reply = infer_test_case_schema(aggregated_user_msg)
    if inferred_schema:
        current_schema = inferred_schema
        st.session_state.test_case_schema = current_schema

    # If the model provided a normal conversational reply (no tool call)
    # we want it to appear in the chat so the user sees the answer.
    if assistant_reply.strip():
        st.session_state.test_case_chat_history.append(
            {"role": "assistant", "content": assistant_reply}
        )

    # --------------------------------------------------------------------
    # 2) Regenerate test cases for affected requirements if needed
    # --------------------------------------------------------------------
    if affected_ids:
        requirements: List[Dict] = st.session_state.get("requirements", [])
        affected_requirements = [r for r in requirements if r["id"] in affected_ids]
        if affected_requirements:
            new_cases = generate_test_cases(
                affected_requirements,
                current_schema,
                instruction,
                st.session_state.get("test_case_sample"),
            )

            # Replace cases for these requirements in the global list
            test_cases = [
                c for c in test_cases if c.get("requirement_id") not in affected_ids
            ]
            test_cases.extend(new_cases)
            st.session_state.test_cases = test_cases
            # Track versions of test cases
            tc_versions = st.session_state.get("test_case_versions", [])
            tc_versions.append([c.copy() for c in test_cases])
            st.session_state["test_case_versions"] = tc_versions
            st.session_state["tc_version_idx"] = len(tc_versions) - 1

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

# Utility to build generic sample test case


def _build_generic_sample(schema: Dict[str, Any]) -> Dict[str, Any]:
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    sample: Dict[str, Any] = {}
    for name in props.keys():
        if name == "requirement_id":
            sample[name] = 1
        elif name == "requirement_name":
            sample[name] = "Sample Requirement"
        else:
            sample[name] = "Sample Value"
    return sample