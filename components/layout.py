"""Page header and section-card layout helpers."""
from contextlib import contextmanager
import streamlit as st


def page_header(icon, title, subtitle=None):
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    html = (
        f'<div class="page-header">'
        f'<div class="page-header-icon">{icon}</div>'
        f'<div class="page-header-text"><h1>{title}</h1>{subtitle_html}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def hero_header(badges, title, quote=None, caption=None):
    """Overview-page hero: pill badges, big title, italic quote, caption."""
    badges_html = "".join(f'<span class="pill {cls}">{text}</span>' for text, cls in badges)
    quote_html = f'<p class="hero-quote">{quote}</p>' if quote else ""
    caption_html = f'<p class="hero-caption">{caption}</p>' if caption else ""
    html = (
        f'<div class="hero-header">'
        f'<div class="header-badges">{badges_html}</div>'
        f'<h1>{title}</h1>'
        f'{quote_html}'
        f'{caption_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def section_title(icon, text):
    """Icon-badge + uppercase caption title, used at the top of a section_card."""
    html = (
        '<div class="section-title">'
        f'<div class="section-title-icon">{icon}</div>'
        f'<div class="section-title-text">{text}</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


@contextmanager
def section_card():
    """Wrap the enclosed Streamlit content in a styled, bordered card container.

    Uses st.container(border=True) rather than raw HTML div tags: Streamlit
    renders each st.markdown call as a separate sibling in the DOM, so an
    open/close pair of markdown divs would not actually nest the widgets
    rendered between them.
    """
    with st.container(border=True):
        yield
