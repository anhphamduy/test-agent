import json
from typing import List, Dict

import streamlit as st
import os

from settings import get_openai_client


# ---------------------------------------------------------------------------
# Requirement extraction
# ---------------------------------------------------------------------------


def extract_requirements(text: str, prompt: str | None = None) -> List[Dict]:
    """Call OpenAI to extract a list of software requirements from raw text.

    Parameters
    ----------
    text : str
        The raw document text from which to extract requirements.
    prompt : str | None
        Custom user-supplied instructions that will be sent as the user message. If
        None, a sensible default instruction is used.
    """
    client = get_openai_client()

    # -----------------------------------------------------------
    # We now ask the LLM to return a **hierarchical** list of
    # viewpoints (3-level MECE structure) each containing an array
    # of requirements. The old flat schema is kept internally by
    # flattening the nested response so downstream steps remain
    # backwards-compatible (they expect a list of requirements with
    # id + name).
    # -----------------------------------------------------------

    lang = "English"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_requirements",
                "description": (
                    "Return a list of viewpoints, each with 3 MECE levels and an array of requirements."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "viewpoints": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "first_level_viewpoint": {
                                        "type": "string",
                                        "description": f"Split the document into user journey steps or modules following MECE principles. Written in {lang}",
                                    },
                                    "second_level_viewpoint": {
                                        "type": "string",
                                        "description": f"Second-level viewpoint grouping. Written in {lang}",
                                    },
                                    "third_level_viewpoint": {
                                        "type": "string",
                                        "description": f"Specific categories, subcategories, or test objectives. Written in {lang}",
                                    },
                                    "requirements": {
                                        "type": "array",
                                        "description": (
                                            "A list of requirements in the viewpoint. CONDITIONS ARE NOT REQUIREMENTS, regardless of any estimate."
                                        ),
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "number": {
                                                    "type": "string",
                                                    "description": "The requirement number",
                                                },
                                                "name": {
                                                    "type": "string",
                                                    "description": (
                                                        f"The name of the requirement with the correct intent. It must be independent – do not reference other requirements. Written in {lang}"
                                                    ),
                                                },
                                            },
                                            "required": ["name"],
                                        },
                                    },
                                },
                                "required": [
                                    "first_level_viewpoint",
                                    "second_level_viewpoint",
                                    "third_level_viewpoint",
                                    "requirements",
                                ],
                            },
                        }
                    },
                    "required": ["viewpoints"],
                },
            },
        }
    ]

    default_instruction = (
        "From the document below, create a MECE hierarchy of viewpoints (3 levels) and list the requirements inside each viewpoint. "
        "Return the result strictly via the extract_requirements function, following the JSON schema provided in the function definition."
    )

    user_instruction = prompt.strip() if prompt else default_instruction

    messages = [
        {
            "role": "system",
            "content": "You are an expert business analyst extracting software requirements.",
        },
        {
            "role": "user",
            "content": user_instruction,
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

    # -----------------------------------------------------------
    # Parse the hierarchical response and flatten it so that each
    # requirement becomes an object with a unique numerical id.
    # -----------------------------------------------------------

    flat_requirements: List[Dict] = []

    if msg.tool_calls:
        for call in msg.tool_calls:
            if call.function.name != "extract_requirements":
                continue

            try:
                args = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                st.warning("Failed to parse requirements from model response.")
                continue

            viewpoints = args.get("viewpoints", []) if isinstance(args, dict) else []

            for vp in viewpoints:
                if not isinstance(vp, dict):
                    continue

                first_vp = vp.get("first_level_viewpoint", "")
                second_vp = vp.get("second_level_viewpoint", "")
                third_vp = vp.get("third_level_viewpoint", "")
                for req in vp.get("requirements", []):
                    if not isinstance(req, dict):
                        continue

                    flat_requirements.append(
                        {
                            "id": len(flat_requirements) + 1,
                            "name": req.get("name", ""),
                            "first_level_viewpoint": first_vp,
                            "second_level_viewpoint": second_vp,
                            "third_level_viewpoint": third_vp,
                        }
                    )

    return flat_requirements 