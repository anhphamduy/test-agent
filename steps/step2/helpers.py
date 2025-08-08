import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import streamlit as st

from constants import DEFAULT_TEST_CASE_ITEM_SCHEMA, DEFAULT_REQUIREMENT_ITEM_SCHEMA
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

# -----------------------------------------
# Generic sample builder for requirement objects
# -----------------------------------------


def _build_generic_requirement_sample(schema: Dict) -> Dict:
    """Return a generic sample requirement following the provided item-level schema."""

    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    sample: Dict = {}
    for name in props.keys():
        if name == "id":
            sample[name] = 1
        elif name == "name":
            sample[name] = "Sample Requirement"
        else:
            sample[name] = "Sample Value"
    return sample


# -----------------------------------------
# Schema inference helper (similar to Step-3)
# -----------------------------------------


def infer_requirement_schema(user_msg: str) -> Tuple[Dict, str]:
    """Detect if the requirement item-level schema needs changes and return it.

    Parameters
    ----------
    user_msg: str
        Aggregated conversation text (including historical messages) ending with the latest user request.

    Returns
    -------
    schema_obj: Dict
        Parsed JSON object of the new schema, or an empty dict if unchanged.
    assistant_reply: str
        Normal chat reply from the assistant when no schema change is required.
    """

    client = get_openai_client()

    current_schema = st.session_state.get(
        "requirement_schema", DEFAULT_REQUIREMENT_ITEM_SCHEMA
    )
    current_requirements = st.session_state.get("requirements", [])

    tools = [
        {
            "type": "function",
            "function": {
                "name": "detect_requirement_schema",
                "description": "Analyse the user's request and decide whether the requirement schema must change.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Short rationale for the decision.",
                        },
                        "schema": {
                            "type": "string",
                            "description": "JSON string of the *full* new requirement item-level schema. Use '{}' if unchanged.",
                        },
                        "sample_requirement": {
                            "type": "string",
                            "description": "JSON string of ONE sample requirement object complying with the schema. Use '{}' if unchanged.",
                        },
                    },
                    "required": ["reason", "schema", "sample_requirement"],
                },
            },
        }
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior business analyst and JSON-Schema expert. Decide whether the item-level "
                "schema for software requirements must change based on the user's request. Respond ONLY by "
                "calling the detect_requirement_schema function **when** a schema change is required. If the "
                "request is purely content editing and no schema adjustment is needed, respond normally without "
                "calling any function. When you do call the function, the `schema` argument must be a JSON string "
                "representing the complete item-level schema (not a patch). All properties should use type 'string' "
                "except for the identifier which may be 'integer'. If you respond normally without calling the function, end your message with a question asking if the user would like to apply these ideas to the requirements table. "
            ),
        },
        {"role": "system", "content": f"Current schema: {json.dumps(current_schema)}"},
        {"role": "system", "content": f"Current requirements: {json.dumps(current_requirements)}"},
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

        # If no tool call, return content as normal assistant reply
        if not assistant_msg.tool_calls:
            return {}, assistant_msg.content or ""

        first_call = assistant_msg.tool_calls[0]
        try:
            args = json.loads(first_call.function.arguments)
            schema_str = args.get("schema", "{}")
            sample_str = args.get("sample_requirement", "{}")

            try:
                schema_obj = json.loads(schema_str) if isinstance(schema_str, str) else {}
            except json.JSONDecodeError:
                schema_obj = {}

            try:
                sample_obj = json.loads(sample_str) if isinstance(sample_str, str) else {}
            except json.JSONDecodeError:
                sample_obj = {}

            # Store sample for UI consistency
            if sample_obj:
                st.session_state["requirement_sample"] = sample_obj
            else:
                if schema_obj:
                    st.session_state["requirement_sample"] = _build_generic_requirement_sample(
                        schema_obj
                    )

            return schema_obj if isinstance(schema_obj, dict) else {}, ""
        except Exception:
            return {}, ""
    except Exception:
        return {}, ""


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

    # -----------------------------------------------------------
    # 1) Schema inference (allows adding/removing fields like Step-3)
    # -----------------------------------------------------------

    # Build aggregated conversation text (including previous messages)
    conversation_lines = [
        f"{m.get('role', '').capitalize()}: {m.get('content', '')}" for m in st.session_state.chat_history
    ]
    conversation_lines.append(f"User: {user_msg}")
    aggregated_user_msg = "\n".join(conversation_lines)

    inferred_schema, assistant_reply = infer_requirement_schema(aggregated_user_msg)
    if inferred_schema:
        st.session_state["requirement_schema"] = inferred_schema


    # Show normal assistant reply if no function was invoked
    if assistant_reply.strip():
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})

    # Refresh local variables after possible schema update
    client = get_openai_client()
    requirements = st.session_state.requirements
    document_text = st.session_state.get("document_text", "")

    # Build item-level schema based on current requirement schema and make every property required
    item_schema: Dict = json.loads(json.dumps(st.session_state.get("requirement_schema", DEFAULT_REQUIREMENT_ITEM_SCHEMA)))  # deep copy
    all_props = list(item_schema.get("properties", {}).keys())
    item_schema["required"] = all_props

    tools = [
        {
            "type": "function",
            "function": {
                "name": "update_requirements",
                "description": "Replace the entire requirements list with a new list provided by the assistant.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "requirements": {"type": "array", "items": item_schema},
                        "summary": {
                            "type": "string",
                            "description": "Brief summary of changes made to the requirements based on user request.",
                        },
                    },
                    "required": ["requirements", "summary"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "trigger_test_case_generation",
                "description": "Generate test cases for selected requirements and advance to Step 3.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "requirement_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "IDs of requirements to generate test cases for. Use an empty list or omit to include all.",
                        },
                        "new_schema": {
                            "type": "string",
                            "description": "JSON string of the full item-level schema to use for generation. Use '{}' to keep current schema.",
                        },
                    },
                    "required": ["requirement_ids", "new_schema"],
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
                    "Whenever the user requests a change, respond by either:\n" \
                    "1) Calling the update_requirements function with the full updated list (plus summary) when editing requirements, OR\n" \
                    "2) Calling the trigger_test_case_generation function when the user asks to generate test cases for some or all requirements. The function **must include** a `requirement_ids` array and a `new_schema` string (use '{}' if unchanged).\n" \
                    "If no change is needed, respond normally **and end your reply with a question asking if the user wants to apply these ideas to the requirements table** (e.g. 'Would you like me to update the requirements accordingly?'). "
                    "Important: If the user asks to delete a requirement that does not exist in the current list, do NOT call any function. "
                    "Instead, reply normally with an apology (e.g. 'Sorry, requirement X does not exist.') so the user is informed."
                    "When you talk about a 'viewpoint', make it clear in your reply that each viewpoint corresponds to an individual requirement from the user's perspective. "
                    "If the user explicitly requests schema changes (e.g. adding a new field), ensure all requirements include that field according to the *current* schema stored in memory. "
                ),
            },
            {
                "role": "system",
                "content": f"Current requirement schema: {json.dumps(st.session_state.get('requirement_schema', DEFAULT_REQUIREMENT_ITEM_SCHEMA))}",
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
    print(response)

    assistant_msg = response.choices[0].message
    assistant_content = assistant_msg.content or ""
    if assistant_content.strip():
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_content})

    if assistant_msg.tool_calls:
        for call in assistant_msg.tool_calls:
            if call.function.name == "trigger_test_case_generation":
                # Parse arguments
                try:
                    args = json.loads(call.function.arguments)
                except Exception:
                    args = {}

                req_ids = args.get("requirement_ids", [])
                new_schema_str = args.get("new_schema", "{}")

                # Determine which requirements to generate for
                if req_ids:
                    selected_reqs = [r for r in st.session_state.requirements if r.get("id") in req_ids]
                else:
                    selected_reqs = st.session_state.requirements

                # Determine base schema: either user-provided new_schema or current schema
                try:
                    incoming_schema = json.loads(new_schema_str) if isinstance(new_schema_str, str) else {}
                except json.JSONDecodeError:
                    incoming_schema = {}

                if incoming_schema:
                    item_schema = incoming_schema
                else:
                    item_schema = json.loads(
                        json.dumps(st.session_state.get("test_case_schema", DEFAULT_TEST_CASE_ITEM_SCHEMA))
                    )

                # Ensure all properties required
                item_schema["required"] = list(item_schema.get("properties", {}).keys())

                # Persist possibly updated schema
                st.session_state["test_case_schema"] = item_schema

                prompt = "Generate 3-5 test cases for each requirement following the specified schema."

                with st.spinner("Generating test cases…"):
                    st.session_state.test_cases = generate_test_cases(
                        selected_reqs,
                        item_schema,
                        prompt,
                        st.session_state.get("test_case_sample"),
                    )

                    # Track versions of test cases
                    tc_versions = st.session_state.get("test_case_versions", [])
                    tc_versions.append([c.copy() for c in st.session_state.test_cases])
                    st.session_state["test_case_versions"] = tc_versions
                    st.session_state["tc_version_idx"] = len(tc_versions) - 1

                # Immediately rerun so main() moves to Step-3 (test-case view)
                st.rerun()
            else:
                requirements, summary = apply_tool_call(requirements, call)
                if summary:
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": f"**Summary of changes:** {summary}",
                        }
                    )
                st.session_state.requirements = requirements
                # Track versions of requirements
                req_versions = st.session_state.get("requirements_versions", [])
                req_versions.append([r.copy() for r in requirements])
                st.session_state["requirements_versions"] = req_versions
                st.session_state["req_version_idx"] = len(req_versions) - 1


# ---------------------------------------------------------------------------
# Test-case generation helpers
# ---------------------------------------------------------------------------

# --------------------------
# Utility to build a generic sample test case given a schema
# --------------------------


def _build_generic_sample(schema: Dict) -> Dict:
    """Return a generic sample object matching the provided item-level schema."""
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    sample: Dict = {}
    for name, definition in props.items():
        if name == "requirement_id":
            sample[name] = 1
        elif name == "requirement_name":
            sample[name] = "Sample Requirement"
        else:
            sample[name] = "Sample Value"
    return sample


# passing the user's current chat message for additional context
def _generate_test_cases_for_requirement(
    requirement: Dict,
    item_schema: Dict,
    user_msg: str = "",
    sample_case: Dict | None = None,
) -> List[Dict]:
    """Internal helper to call OpenAI and generate test cases for a single requirement."""

    client = get_openai_client()

    # Ensure every property is required
    full_item_schema = json.loads(json.dumps(item_schema))
    full_item_schema["required"] = list(full_item_schema.get("properties", {}).keys())

    tools = [
        {
            "type": "function",
            "function": {
                "name": "generate_test_cases",
                "description": (
                    "Return manual test cases for the given requirement, strictly following the provided JSON schema."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "test_cases": {"type": "array", "items": full_item_schema}
                    },
                    "required": ["test_cases"],
                },
            },
        }
    ]

    # Build system & user messages
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
                "Generate test cases for the requirement below and return them **only** via the "
                "generate_test_cases function.\n\n"
                + f"Requirement: {requirement['name']}"
            ),
        },
    ]

    # Provide sample case as additional guidance for formatting consistency
    if sample_case is None:
        sample_case = _build_generic_sample(item_schema)

    messages.insert(
        1,
        {
            "role": "system",
            "content": (
                "Here is an example test case that strictly follows the schema. Model outputs should be consistent with this format:\n"
                f"```json\n{json.dumps(sample_case)}\n```"
            ),
        },
    )

    # Inject the latest user query as extra context for the LLM if provided
    if user_msg:
        messages.insert(
            1,
            {
                "role": "user",
                "content": user_msg,
            },
        )

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


# Accept optional user message to propagate down to each test-case generation call


def generate_test_cases(
    requirements: List[Dict],
    item_schema: Dict,
    user_msg: str = "",
    sample_case: Dict | None = None,
) -> List[Dict]:
    """Generate test cases for each requirement using parallel OpenAI calls."""

    if not requirements:
        return []

    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=min(20, len(requirements))) as executor:
        future_to_req = {
            executor.submit(
                _generate_test_cases_for_requirement,
                r,
                item_schema,
                user_msg,
                sample_case,
            ): r
            for r in requirements
        }

        for future in as_completed(future_to_req):
            try:
                results.extend(future.result())
            except Exception:
                # Ignore failures for individual requirements
                continue

    return results