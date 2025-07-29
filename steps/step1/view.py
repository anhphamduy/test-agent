import streamlit as st

from .helpers import extract_requirements


def render():
    """Render Step 1 with a centered uploader, preview, and cleaner styling."""
    

    # Center the uploader & button using a 3-column trick
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.title("📄 Requirement Extractor & Editor")
        st.header("Step 1: Upload Document")
        txt_file = st.file_uploader(
            "Upload a .txt or .md file containing requirements", type=["txt", "md"]
        )

        # (Prompt editing removed to simplify the interface)

        if txt_file:
            bytes_data = txt_file.getvalue()

            # Full document preview so users can inspect the entire content
            with st.expander("📄 Document Preview"):
                st.markdown(bytes_data.decode("utf-8", errors="ignore"))

            if st.button("Extract Requirements", type="primary", use_container_width=True):
                raw_text = bytes_data.decode("utf-8", errors="ignore")
                with st.spinner("Extracting requirements with OpenAI..."):
                    st.session_state.document_text = raw_text
                    st.session_state.requirements = extract_requirements(raw_text)
                st.success(
                    f"✅ Extracted {len(st.session_state.requirements)} requirements!"
                )
                st.rerun() 