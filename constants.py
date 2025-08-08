from typing import Dict

# Central location for constants used across the Streamlit app
# ----------------------------------------------------------

# Default JSON-schema describing the structure of each manual test-case item
# This is shared by Step-2 (generation) and Step-3 (editing) helpers so that
# both steps stay in sync when validating test-case data.

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

DEFAULT_REQUIREMENT_ITEM_SCHEMA: Dict = {
    "type": "object",
    "properties": {
        "id": {
            "type": "integer",
            "description": "Unique identifier of the requirement, starting at 1.",
        },
        "name": {
            "type": "string",
            "description": "Short requirement description.",
        },
        "first_level_viewpoint": {
            "type": "string",
            "description": "Top-level viewpoint (e.g. user journey step or module).",
        },
        "second_level_viewpoint": {
            "type": "string",
            "description": "Second-level grouping or perspective.",
        },
        "third_level_viewpoint": {
            "type": "string",
            "description": "Fine-grained sub-category or test objective.",
        },
    },
    "required": [
        "id",
        "name",
        "first_level_viewpoint",
        "second_level_viewpoint",
        "third_level_viewpoint",
    ],
} 