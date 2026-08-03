"""Persistent top-right university/partner logo strip for the main content area."""
import os
import streamlit as st

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
WREXHAM_LOGO = os.path.join(_ASSETS_DIR, "wrexham_logo.png")
NIBSS_LOGO   = os.path.join(_ASSETS_DIR, "nibss_logo.png")


def render_topbar():
    """Render the logo strip. Uses real st.columns/st.image (not a raw HTML
    wrapper div) — Streamlit renders each st.markdown call as its own
    sibling in the DOM, so a markdown div wrapped around st.image widgets
    would not actually nest them (see components/layout.py's section_card
    docstring for the same gotcha).
    """
    if not (os.path.exists(WREXHAM_LOGO) or os.path.exists(NIBSS_LOGO)):
        return
    _, right = st.columns([3, 2])
    with right:
        c1, c2 = st.columns(2)
        if os.path.exists(WREXHAM_LOGO):
            c1.image(WREXHAM_LOGO, use_container_width=True)
        if os.path.exists(NIBSS_LOGO):
            c2.image(NIBSS_LOGO, use_container_width=True)
