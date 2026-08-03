"""Shared brand palette and CSS injection for the premium banking theme."""
import os
import streamlit as st

NAVY  = "#0D2240"
NAVY_LIGHT = "#1A3A63"
GREEN = "#0F6B5B"
GOLD  = "#D4A72C"
BG    = "#F4F6F9"
CARD_BG = "#FFFFFF"
BORDER  = "#E3E8EF"
TEXT_MUTED = "#5B6B82"

CHART_PALETTE = {
    "primary": NAVY,
    "accent": GOLD,
    "positive": GREEN,
    "negative": "#B3261E",
    "neutral": "#AAB4C2",
    "grid": BORDER,
}

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")


@st.cache_resource(show_spinner=False)
def _read_css():
    css_path = os.path.join(_ASSETS_DIR, "styles.css")
    with open(css_path) as f:
        return f.read()


def inject_css():
    """Inject the shared stylesheet once per session."""
    st.markdown(f"<style>{_read_css()}</style>", unsafe_allow_html=True)
