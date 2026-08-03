"""Reusable app footer."""
import streamlit as st


def render_footer():
    st.markdown("---")
    st.markdown(
        '<p class="app-footer">COM 752 MSc Dissertation · Otu Samuel Jacob · '
        "s25007038 · Wrexham University · 2025/2026</p>",
        unsafe_allow_html=True,
    )
