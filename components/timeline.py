"""Numbered timeline component (used for the framework step list)."""
import streamlit as st


def timeline(steps):
    """Render a connected numbered timeline.

    `steps` is a list of dicts: {"title": str, "desc": str, "icon": str (optional)}.
    """
    parts = ['<div class="timeline">']
    for i, step in enumerate(steps, start=1):
        icon_html = f'<div class="timeline-icon">{step.get("icon", "")}</div>' if step.get("icon") else ""
        parts.append(
            '<div class="timeline-item">'
            '<div class="timeline-connector"></div>'
            f'<div class="timeline-step">{i}</div>'
            '<div class="timeline-body">'
            f'<p class="timeline-title">{step["title"]}</p>'
            f'<p class="timeline-desc">{step["desc"]}</p>'
            '</div>'
            f'{icon_html}'
            '</div>'
        )
    parts.append('</div>')
    st.markdown(''.join(parts), unsafe_allow_html=True)
