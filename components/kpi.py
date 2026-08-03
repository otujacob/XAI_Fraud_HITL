"""Premium KPI card component: icon-in-circle badge + value + caption."""
import streamlit as st

_ACCENTS = {"green", "blue", "purple", "gold", "red", "navy"}


def kpi_card(label, value, delta=None, icon="", accent="green"):
    """Render a single styled KPI card in the current Streamlit container."""
    accent_cls = f"accent-{accent}" if accent in _ACCENTS else "accent-green"
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    html = (
        f'<div class="kpi-card">'
        f'<div class="kpi-icon {accent_cls}">{icon}</div>'
        f'<div class="kpi-body">'
        f'<div class="kpi-top">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{delta_html}'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def kpi_row(items):
    """Render a row of KPI cards. `items` is a list of dicts matching kpi_card's kwargs."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            kpi_card(**item)
