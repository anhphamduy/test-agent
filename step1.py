from typing import List, Dict

import streamlit as st

from helpers import extract_requirements


def render():
    """Render Step 1: file upload and requirement extraction."""
    st.header("Step 1: Upload Document")

    txt_file = st.file_uploader("Upload a .txt file containing requirements", type=["txt"])

    if txt_file and st.button("Extract Requirements", type="primary"):
        raw_text = txt_file.read().decode("utf-8", errors="ignore")
        with st.spinner("Extracting requirements with OpenAI..."):
            st.session_state.document_text = raw_text
            st.session_state.requirements = extract_requirements(raw_text)
        st.success(f"✅ Extracted {len(st.session_state.requirements)} requirements!")
        st.rerun() 