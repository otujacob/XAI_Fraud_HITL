"""Decision badge component (BLOCK / REVIEW / APPROVE pills)."""
import streamlit as st

_BADGES = {
    "BLOCK":   ("🔴", "block"),
    "REVIEW":  ("🟡", "review"),
    "APPROVE": ("🟢", "approve"),
}


def decision_badge_html(decision):
    """Return the inline HTML for a decision pill (for embedding in dataframes/markdown)."""
    icon, css_cls = _BADGES.get(decision, ("⚪", ""))
    return f'<span class="decision-badge {css_cls}">{icon} {decision}</span>'


def decision_badge(decision):
    """Render a decision pill directly in the current Streamlit container."""
    st.markdown(decision_badge_html(decision), unsafe_allow_html=True)
