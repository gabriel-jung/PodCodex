"""
Shared UI utilities for podcodex Streamlit tabs.

Only streamlit-specific helpers belong here. Domain logic belongs in src/podcodex/.
"""

from __future__ import annotations

import re

import streamlit as st


def normalize_path(path: str) -> str:
    """Normalize a user-provided path for use on WSL.

    Handles:
    - Shell-escaped characters (e.g. /mnt/d/My\ Folder\&\ Stuff -> /mnt/d/My Folder& Stuff)
    - Windows paths (e.g. C:\\Users\\gabriel -> /mnt/c/Users/gabriel)
    - Surrounding quotes
    """
    path = path.strip().strip("'\"")

    # Windows path: convert to WSL /mnt/<drive>/...
    win_match = re.match(r"^([A-Za-z]):[/\\](.*)", path)
    if win_match:
        drive = win_match.group(1).lower()
        rest = win_match.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{rest}"

    # Unescape shell escape sequences (backslash followed by any character)
    path = re.sub(r"\\(.)", r"\1", path)

    return path


def fmt_time(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS (e.g. 4356 → '1:12:36')."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


# ── LLM provider presets ─────────────────────

PROVIDERS = {
    "Mistral": {"url": "https://api.mistral.ai/v1", "model": "mistral-small-latest"},
    "OpenAI": {"url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "Custom": {"url": "", "model": ""},
}


def on_provider_change(prefix: str) -> None:
    """Sync model/URL session state when the provider selectbox changes.

    Args:
        prefix: session state key prefix (e.g. "polish" or "trad").
    """
    provider = st.session_state.get(f"{prefix}_api_provider", "Mistral")
    preset = PROVIDERS.get(provider, {})
    if preset["url"]:
        st.session_state[f"{prefix}_api_base_url"] = preset["url"]
        st.session_state[f"{prefix}_api_model"] = preset["model"]


def build_llm_kwargs(prefix: str, mode: str, **extra) -> dict:
    """Build kwargs dict for polish_segments / translate_segments.

    Args:
        prefix: session state key prefix (e.g. "polish" or "trad").
        mode: "api", "ollama", or "manual".
        **extra: additional kwargs passed through (e.g. source_lang, target_lang).
    """
    kwargs = dict(mode=mode, **extra)
    if mode == "api":
        kwargs["model"] = st.session_state.get(
            f"{prefix}_api_model", PROVIDERS["Mistral"]["model"]
        )
        kwargs["api_base_url"] = st.session_state.get(
            f"{prefix}_api_base_url", PROVIDERS["Mistral"]["url"]
        )
        api_key = st.session_state.get(f"{prefix}_api_key_input", "").strip()
        if api_key:
            kwargs["api_key"] = api_key
    elif mode == "ollama":
        kwargs["model"] = st.session_state.get(f"{prefix}_ollama_model", "qwen3:14b")
    return kwargs
