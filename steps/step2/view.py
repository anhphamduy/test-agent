import pandas as pd
import streamlit as st

from .helpers import handle_chat, generate_test_cases


def render():
    """Render Step 2: review, refine, and chat-based editing of requirements."""
    st.header("Step 2: Review and Refine Requirements")

    # Two columns with a wider left side for the table
    col_left, col_right = st.columns([3, 2])

    # Left column: requirements table
    with col_left:

        # Pagination control (subheader removed per request)

        # ------------------------
        # Version navigation UI for requirements
        # ------------------------
        if "requirements_versions" not in st.session_state:
            # Create a placeholder version (empty or current requirements if any)
            st.session_state.requirements_versions = [
                [r.copy() for r in st.session_state.get("requirements", [])]
            ]
            st.session_state.req_version_idx = 0

        if "requirements_versions" in st.session_state:
            total_versions = len(st.session_state.requirements_versions)
            if "req_version_idx" not in st.session_state:
                st.session_state.req_version_idx = total_versions - 1

            idx = st.session_state.req_version_idx

            nav_prev, nav_label, nav_next = st.columns([1, 3, 1])

            with nav_prev:
                if st.button("⬅️", disabled=idx <= 0, use_container_width=True, key="req_prev"):
                    st.session_state.req_version_idx = max(0, idx - 1)
                    st.session_state.requirements = st.session_state.requirements_versions[
                        st.session_state.req_version_idx
                    ]
                    st.rerun()

            with nav_label:
                st.markdown(
                    f"<h3 style='text-align:center; padding-top:2px;'>Version {idx + 1} / {total_versions}</h3>",
                    unsafe_allow_html=True,
                )

            with nav_next:
                if st.button("➡️", disabled=idx >= total_versions - 1, use_container_width=True, key="req_next"):
                    st.session_state.req_version_idx = min(total_versions - 1, idx + 1)
                    st.session_state.requirements = st.session_state.requirements_versions[
                        st.session_state.req_version_idx
                    ]
                    st.rerun()

        # Sort by requirement ID for easier readability
        df = pd.DataFrame(st.session_state.get("requirements", []))

        # Ensure the ID and viewpoint columns are displayed first if they exist
        preferred_order = [
            "id",  # requirement identifier
            "first_level_viewpoint",
            "second_level_viewpoint",
            "third_level_viewpoint",
        ]
        existing_pref_cols = [c for c in preferred_order if c in df.columns]
        other_cols = [c for c in df.columns if c not in existing_pref_cols]
        df = df[existing_pref_cols + other_cols]

        # Sort by requirement ID (if present) for easier readability
        if "id" in df.columns:
            df = df.sort_values(by="id", ascending=True)

        df = df.reset_index(drop=True)
        st.dataframe(df, use_container_width=True, hide_index=True, height=650)


        # Default prompt & schema for test-case generation
        from constants import DEFAULT_TEST_CASE_ITEM_SCHEMA
        default_test_case_prompt = "Generate 3-5 test cases for each requirement following the specified schema."

        # Action to generate test cases directly beneath the table
        if st.button(
            "🚀 Generate Test Cases for All Requirements",
            type="primary",
            use_container_width=True,
        ):

            # Always use default schema
            st.session_state.test_case_schema = DEFAULT_TEST_CASE_ITEM_SCHEMA.copy()

            with st.spinner(
                "Generating test cases in parallel. This may take a while for large requirement sets..."
            ):
                st.session_state.test_cases = generate_test_cases(
                    st.session_state.requirements,
                    DEFAULT_TEST_CASE_ITEM_SCHEMA,
                    default_test_case_prompt,
                    st.session_state.get("test_case_sample"),
                )
                # Store version for test cases
                tc_versions = st.session_state.get("test_case_versions", [])
                tc_versions.append([c.copy() for c in st.session_state.test_cases])
                st.session_state["test_case_versions"] = tc_versions
                st.session_state["tc_version_idx"] = len(tc_versions) - 1
            st.success(f"Generated {len(st.session_state.test_cases)} test cases!")
            st.rerun()

    # Right column: chat interface
    with col_right:

        chat_container = st.container(height=706)

        # Display chat history
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # If no previous chat messages, display a friendly welcome/instruction
            if not st.session_state.chat_history:
                with st.chat_message("assistant"):
                    st.markdown(
                        "👋 Hello! I can help you refine your requirements. Ask me to add, remove, or modify items, and I'll propose updates accordingly."
                    )

        # Chat input
        user_input = st.chat_input("Ask me to add, remove, or modify requirements...")

        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Display user message immediately
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_input)

            # Let OpenAI handle the update in the background
            with chat_container:
                with st.spinner("🤔 Processing your request..."):
                    handle_chat(user_input)

            st.rerun() 