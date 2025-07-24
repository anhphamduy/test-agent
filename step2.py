import pandas as pd
import streamlit as st

from helpers import handle_chat


def render():
    """Render Step 2: review, refine, and chat-based editing of requirements."""
    st.header("Step 2: Review and Refine Requirements")

    # Two equal-width columns for requirements table and chat interface
    col_left, col_right = st.columns(2)

    # Left column: requirements table
    with col_left:
        st.subheader("📋 Current Requirements")
        df = pd.DataFrame(st.session_state.requirements)
        st.dataframe(df, use_container_width=True, hide_index=True, height=600)

    # Right column: chat interface
    with col_right:
        st.subheader("💬 Chat to Modify Requirements")

        chat_container = st.container(height=600)

        # Display chat history
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

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

    # Divider and action to move to test case generation
    st.divider()

    from helpers import generate_test_cases

    if st.button("🚀 Generate Test Cases for All Requirements", type="primary"):
        with st.spinner("Generating test cases in parallel. This may take a while for large requirement sets..."):
            st.session_state.test_cases = generate_test_cases(
                st.session_state.requirements,
                st.session_state.get("test_case_schema", {}),
            )
        st.success(f"Generated {len(st.session_state.test_cases)} test cases!")
        st.rerun() 