import os
import json
import re
from typing import List, Dict, Tuple

import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# --------------------------
# Default schemas
# --------------------------

DEFAULT_TEST_CASE_ITEM_SCHEMA: Dict = {
    "type": "object",
    "properties": {
        "requirement_id": {
            "type": "integer",
            "description": "Identifier of the requirement this test case is linked to.",
        },
        "requirement_name": {
            "type": "string",
            "description": "Concise name of the originating requirement.",
        },
        "test_case_name": {
            "type": "string",
            "description": "Title or identifier of the test case.",
        },
        "test_steps": {
            "type": "string",
            "description": "Step-by-step instructions in natural language.",
        },
        "expected_result": {
            "type": "string",
            "description": "Outcome that should be observed when the steps are executed successfully.",
        },
    },
    "required": [
        "requirement_id",
        "requirement_name",
        "test_case_name",
        "test_steps",
        "expected_result",
    ],
}

# --------------------------
# Session-state utilities
# --------------------------


def init_session_state():
    """Ensure required keys exist in st.session_state."""
    if "requirements" not in st.session_state:
        st.session_state.requirements: List[Dict] = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[Dict] = []
    if "document_text" not in st.session_state:
        st.session_state.document_text: str = ""
    if "test_cases" not in st.session_state:
        st.session_state.test_cases: List[Dict] = []
    if "test_case_chat_history" not in st.session_state:
        st.session_state.test_case_chat_history: List[Dict] = []
    if "test_case_schema" not in st.session_state:
        # Make a shallow copy to avoid accidental global mutation
        st.session_state.test_case_schema: Dict = DEFAULT_TEST_CASE_ITEM_SCHEMA.copy()


# --------------------------
# Azure OpenAI helpers
# --------------------------


def get_openai_client() -> AzureOpenAI:
    """Instantiate an Azure OpenAI client using env vars."""

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    if not api_key or not endpoint:
        st.error(
            "Missing Azure OpenAI configuration. Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT."
        )
        st.stop()

    return AzureOpenAI(
        azure_endpoint=endpoint, api_key=api_key, api_version=api_version
    )


# --------------------------
# Requirement extraction & manipulation
# --------------------------


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
    return []  # Fallback


def next_requirement_id(requirements: List[Dict]) -> int:
    return max((r["id"] for r in requirements), default=0) + 1


def apply_tool_call(requirements: List[Dict], call) -> Tuple[List[Dict], str]:
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
        st.session_state.chat_history.append(
            {"role": "assistant", "content": assistant_content}
        )

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


# --------------------------
# Test case generation
# --------------------------


def _generate_test_cases_for_requirement(
    requirement: Dict, item_schema: Dict
) -> List[Dict]:
    """Internal helper to call OpenAI and generate test cases for a single requirement.

    The returned test case objects must conform to `item_schema`. To keep flexibility, we
    embed the JSON schema verbatim into the system prompt and rely on the model to output
    data that matches it via the `generate_test_cases` function. We also dynamically set
    the `items` schema of the tool definition so that the function-call response is
    validated by the OpenAI tooling layer.
    """
    # Debug prints removed
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
                        # Ensure the correct linkage regardless of what the model returned
                        c["requirement_id"] = requirement["id"]
                        c["requirement_name"] = requirement["name"]
                        cases.append(c)
                except json.JSONDecodeError:
                    pass  # Skip badly formatted responses

    return cases


def generate_test_cases(requirements: List[Dict], item_schema: Dict) -> List[Dict]:
    """Generate test cases for each requirement using parallel OpenAI calls.

    Each requirement triggers a separate chat completion call executed concurrently. The
    returned dictionaries must conform to `item_schema` and will have `requirement_id` &
    `requirement_name` populated automatically.
    """
    
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


# --------------------------
# Schema inference for dynamic test case structure
# --------------------------


def infer_test_case_schema(user_msg: str) -> Tuple[Dict, List[int]]:
    """Determine schema changes and affected requirements via OpenAI function calling.

    We expose a single function `detect_schema_and_affected` and *force* the model to
    call it every time. The function parameters:

    {
        "schema": object – new item-level schema or empty object if unchanged,
        "affected_requirement_ids": array<number> – IDs whose test cases must be regenerated,
        "reason": string – optional free-text rationale (ignored by code)
    }
    """

    client = get_openai_client()

    # Context for the LLM
    current_schema = st.session_state.get("test_case_schema", DEFAULT_TEST_CASE_ITEM_SCHEMA)
    current_test_cases = st.session_state.get("test_cases", [])
    requirements = st.session_state.get("requirements", [])

    # Define the function schema for tool calling
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
                    "required": ["schema", "affected_requirement_ids", "reason"],
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
                "representing the full item-level schema, or '{}' if unchanged."
            ),
        },
        {"role": "system", "content": f"Current schema: {json.dumps(current_schema)}"},
        {"role": "system", "content": f"Requirements list: {json.dumps(requirements)}"},
        {"role": "system", "content": f"Existing test cases: {json.dumps(current_test_cases)}"},
        {"role": "user", "content": user_msg},
    ]

    try:
        resp = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-35-turbo"),
            messages=messages,
            tools=tools,
            # Force the function call every time
            tool_choice={"type": "function", "function": {"name": "detect_schema_and_affected"}},
            temperature=0.0,
        )

        assistant_msg = resp.choices[0].message
        if not assistant_msg.tool_calls:
            return {}, []  # Should not happen, but be safe

        first_call = assistant_msg.tool_calls[0]
        try:
            args = json.loads(first_call.function.arguments)
            schema_str = args.get("schema", "{}")
            try:
                schema_obj = json.loads(schema_str) if isinstance(schema_str, str) else {}
            except json.JSONDecodeError:
                schema_obj = {}
            affected_ids = args.get("affected_requirement_ids", [])

            if not isinstance(schema_obj, dict):
                schema_obj = {}
            if not isinstance(affected_ids, list):
                affected_ids = []

            affected_ids = [int(i) for i in affected_ids if isinstance(i, (int, float, str)) and str(i).isdigit()]

            return schema_obj, affected_ids
        except Exception:
            return {}, []
    except Exception:
        return {}, []


# --------------------------
# Test case chat handler
# --------------------------


def apply_test_case_tool_call(test_cases: List[Dict], call) -> Tuple[List[Dict], str]:
    """Handle the update_test_cases function call coming from the model."""

    if call.function.name != "update_test_cases":
        return test_cases, ""

    try:
        args = json.loads(call.function.arguments)
        new_cases = args.get("test_cases", None)
        summary = args.get("summary", "")

        if new_cases is None:
            # Malformed call (missing argument) – ignore
            return test_cases, ""

        # Ensure we only replace when the assistant supplies a non-empty list
        if (
            isinstance(new_cases, list)
            and len(new_cases) > 0
            and all(isinstance(c, dict) for c in new_cases)
        ):
            return new_cases, summary
        # If the model returned an empty list, treat it as no-op to avoid losing existing test cases
        return test_cases, ""
    except json.JSONDecodeError:
        pass
    return test_cases, ""


def handle_test_case_chat(user_msg: str):
    """Process a chat message related to test cases, allowing the model to modify them via function calls."""

    client = get_openai_client()

    test_cases = st.session_state.test_cases

    # ---------------------------------------------------
    # 1) Check if the user request implies a schema change
    # ---------------------------------------------------
    current_schema: Dict = st.session_state.get(
        "test_case_schema", DEFAULT_TEST_CASE_ITEM_SCHEMA
    )
    inferred_schema, affected_ids = infer_test_case_schema(user_msg)
    if inferred_schema:  # Non-empty dict with "properties"
        current_schema = inferred_schema
        st.session_state.test_case_schema = current_schema

    # After potentially updating the schema, determine if the user wants regeneration for
    # specific requirements and perform it so that new test cases conform to `current_schema`.
    # ---------------------------------------------------

    # The `affected_ids` list was returned by the schema inference step above
    if affected_ids:
        requirements: List[Dict] = st.session_state.get("requirements", [])
        affected_requirements = [r for r in requirements if r["id"] in affected_ids]
        if affected_requirements:
            new_cases = generate_test_cases(affected_requirements, current_schema)

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
            # Finished regeneration; skip further LLM processing
            return
