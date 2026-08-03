"""Reusable branded sidebar: app mark, pill navigation, project info, and logos."""
import os
import streamlit as st

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
WREXHAM_LOGO = os.path.join(_ASSETS_DIR, "wrexham_logo.png")
NIBSS_LOGO   = os.path.join(_ASSETS_DIR, "nibss_logo.png")

PAGES = [
    ("🏠", "Overview"),
    ("📊", "Model Results"),
    ("🔍", "SHAP Explanations"),
    ("🔄", "HITL Feedback"),
    ("⚡", "Live Scoring"),
]


def render_sidebar():
    """Render the branded sidebar and return the selected page name."""
    if "page" not in st.session_state:
        st.session_state.page = PAGES[0][1]

    with st.sidebar:
        st.markdown(
            '<div class="sidebar-mark">'
            '<div class="mark-icon">🛡️</div>'
            '<div class="mark-text">'
            '<div class="mark-title">XAI FRAUD<br/>DETECTION</div>'
            '<div class="mark-subtitle">Nigerian Inter-Bank Payments</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<p class="sidebar-label">Navigation</p>', unsafe_allow_html=True)
        for icon, name in PAGES:
            is_active = st.session_state.page == name
            if st.button(
                name,
                icon=icon,
                key=f"nav_{name}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state.page = name
                st.rerun()

        st.markdown('<p class="sidebar-label">Project Info</p>', unsafe_allow_html=True)
        st.markdown(
            '<div class="project-info-box">'
            '<div class="info-field"><p class="info-label">Student</p><p class="info-value">Otu Samuel Jacob</p></div>'
            '<div class="info-field"><p class="info-label">ID</p><p class="info-value">s25007038</p></div>'
            '<div class="info-field"><p class="info-label">University</p><p class="info-value">Wrexham University</p></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.caption("Framework: IF + XGBoost + RF + ADASYN + SHAP")

        st.markdown("---")
        if os.path.exists(WREXHAM_LOGO):
            st.image(WREXHAM_LOGO, use_container_width=True)
        if os.path.exists(NIBSS_LOGO):
            st.image(NIBSS_LOGO, use_container_width=True)

    return st.session_state.page
