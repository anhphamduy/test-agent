import pandas as pd
import streamlit as st

from helpers import handle_test_case_chat


def render():
    """Render Step 3: review, refine, and chat-based editing of test cases."""

    st.header("Step 3: Generated Test Cases")

    if not st.session_state.get("test_cases"):
        st.info("No test cases have been generated yet. Please go back to Step 2 and generate them.")
        return

    # Create two columns like in Step 2: table on left, chat on right
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🧪 Test Cases")

        # Ensure the dataframe includes all fields defined in the current test-case schema
        schema_props = (
            st.session_state.get("test_case_schema", {})
            .get("properties", {})
        )

        df = pd.DataFrame(st.session_state.test_cases)

        # Add any missing columns (initially with empty strings) so users can see new schema fields immediately
        for col in schema_props.keys():
            if col not in df.columns:
                df[col] = ""

        # Re-order columns to match the schema definition order for consistency
        df = df[list(schema_props.keys())] if schema_props else df

        st.dataframe(df, use_container_width=True, hide_index=True, height=600)

        # CSV download button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Test Cases as CSV",
            data=csv,
            file_name="test_cases.csv",
            mime="text/csv",
        )

        # Back button
        if st.button("↩️ Back to Requirements", key="back_to_step2"):
            st.session_state.pop("test_cases", None)
            st.rerun()

    with col_right:
        st.subheader("💬 Chat to Modify Test Cases")

        chat_container = st.container(height=600)

        # Display chat history specific to test cases
        with chat_container:
            for msg in st.session_state.test_case_chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

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