import pandas as pd
import streamlit as st

from .helpers import handle_test_case_chat


def render():
    """Render Step 3: review, refine, and chat-based editing of test cases."""

    st.header("Step 3: Generated Test Cases")

    if "test_cases" not in st.session_state:
        st.session_state.test_cases = []

    # Create two columns like in Step 2 but with wider table column
    col_left, col_right = st.columns([3, 2])

    with col_left:
        # (header removed as per UI simplification)

        # Ensure version list exists so navigation shows at least one version
        if "test_case_versions" not in st.session_state:
            st.session_state.test_case_versions = [
                [c.copy() for c in st.session_state.get("test_cases", [])]
            ]
            st.session_state.tc_version_idx = 0

        # ------------------------
        # Version navigation UI for test cases
        # ------------------------
        if "test_case_versions" in st.session_state:
            total_tc_ver = len(st.session_state.test_case_versions)
            if "tc_version_idx" not in st.session_state:
                st.session_state.tc_version_idx = total_tc_ver - 1

            tc_idx = st.session_state.tc_version_idx

            nav_prev_tc, nav_label_tc, nav_next_tc = st.columns([1,3,1])

            with nav_prev_tc:
                if st.button("⬅️", disabled=tc_idx <= 0, use_container_width=True, key="tc_prev"):
                    st.session_state.tc_version_idx = max(0, tc_idx - 1)
                    st.session_state.test_cases = st.session_state.test_case_versions[
                        st.session_state.tc_version_idx
                    ]
                    st.rerun()

            with nav_label_tc:
                st.markdown(
                    f"<h3 style='text-align:center; padding-top:2px;'>Version {tc_idx + 1} / {total_tc_ver}</h3>",
                    unsafe_allow_html=True,
                )

            with nav_next_tc:
                if st.button("➡️", disabled=tc_idx >= total_tc_ver - 1, use_container_width=True, key="tc_next"):
                    st.session_state.tc_version_idx = min(total_tc_ver - 1, tc_idx + 1)
                    st.session_state.test_cases = st.session_state.test_case_versions[
                        st.session_state.tc_version_idx
                    ]
                    st.rerun()

        # Ensure the dataframe includes all fields defined in the current test-case schema
        schema_props = (
            st.session_state.get("test_case_schema", {})
            .get("properties", {})
        )

        # Build DataFrame and sort by requirement ID if present
        df = pd.DataFrame(st.session_state.test_cases)

        if "requirement_id" in df.columns:
            df = df.sort_values(by="requirement_id", ascending=True).reset_index(drop=True)

        # Add any missing columns so users can see new schema fields immediately
        for col in schema_props.keys():
            if col not in df.columns:
                df[col] = ""

        # Re-order columns to match the schema definition order for consistency
        df = df[list(schema_props.keys())] if schema_props else df

        st.dataframe(df, use_container_width=True, hide_index=True, height=650)

        # CSV download button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Test Cases as CSV",
            data=csv,
            file_name="test_cases.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_right:
        chat_container = st.container(height=706)

        # Display chat history specific to test cases
        with chat_container:
            for msg in st.session_state.test_case_chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # If no previous test-case chat messages, display a friendly welcome/instruction
            if not st.session_state.test_case_chat_history:
                with st.chat_message("assistant"):
                    st.markdown(
                        "👋 Hello! I can help you refine your test cases. Ask me to add, remove, or modify test cases, and I'll propose updates accordingly."
                    )

        # Chat input
        user_input = st.chat_input("Ask me to add, remove, or modify test cases...")

        if user_input:
            # Add user message to history
            st.session_state.test_case_chat_history.append({"role": "user", "content": user_input})

            # Display user message immediately
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_input)

            # Process chat with OpenAI
            with chat_container:
                with st.spinner("🤔 Processing your request..."):
                    handle_test_case_chat(user_input)

            st.rerun() 