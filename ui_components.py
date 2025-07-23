import streamlit as st
import pandas as pd
import re
from openai_client import ask_openai
from config import DEFAULT_MODEL

def render_strategy_cards(df: pd.DataFrame) -> None:
    """
    Render hedge strategies in a single horizontal scrollable row.
    Hovering over the strategy name shows one action: "Select Strategy".
    """
    if df.empty:
        st.info("No strategies generated yet.")
        return

    st.markdown("""<div style='display: flex; overflow-x: auto; gap: 1rem; padding-bottom: 1rem;'>""", unsafe_allow_html=True)

    for i, row in df.iterrows():
        chosen = st.session_state.get("chosen_strategy") or {}
        selected = chosen.get("name") == row.name
        border = "#10b981" if selected else "#60A5FA"

        headline = str(row.rationale.get("thesis", row.rationale))
        short_title = " ".join(headline.split()[:5]) + "…"

        st.markdown(
            f"""
            <style>
            .strategy-card-{i} {{
                min-width: 220px;
                max-width: 220px;
                border: 1px solid {border};
                border-radius: 8px;
                background: #1e293b;
                color: #f8fafc;
                padding: 10px;
                font-size: 13px;
                font-weight: 600;
                position: relative;
                cursor: pointer;
                transition: transform 0.2s ease;
            }}
            .strategy-card-{i}:hover {{
                transform: scale(1.02);
            }}
            .strategy-card-{i} .hover-action-{i} {{
                display: none;
                position: absolute;
                top: 100%;
                left: 0;
                margin-top: 4px;
                padding: 3px 10px;
                background: #1e40af;
                color: white;
                border-radius: 4px;
                font-size: 11px;
            }}
            .strategy-card-{i}:hover .hover-action-{i} {{
                display: block;
            }}
            </style>
            <div class="strategy-card-{i}" onclick="window.location.search='?selected_strategy_idx={i}';">
                {short_title}
                <div class="hover-action-{i}">Select Strategy</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("""</div>""", unsafe_allow_html=True)

def clean_md(md: str) -> str:
    """
    Cleans Markdown text by adding spaces after digits followed by letters
    and removing asterisks and underscores.
    """
    md = re.sub(r"(\d)(?=[A-Za-z])", r"\1 ", md)
    return md.replace("*", "").replace("_", "")

def render_rationale(df: pd.DataFrame) -> None:
    """
    Renders a detailed explanation of allocated hedge instruments and their rationale.
    """
    if df.empty:
        st.info("No hedge rationale to show.")
        return

    total = df["Amount ($)"].sum()
    st.markdown(
        f"A total of **${total:,.0f}** was allocated to hedge instruments "
        "to mitigate key risks in the portfolio.\n\n"
        "Below is the explanation for each hedge component:"
    )

    for _, row in df.iterrows():
        tick = row.get("Ticker", "—").strip()
        pos = row.get("Position", "—").title()
        amt = row.get("Amount ($)", 0)
        rat = row.get("Rationale", "No rationale provided").strip()
        src = row.get("Source", "").strip()

        # HTML card for each hedge component's rationale
        card = (
            f"<div style='background:#1e293b;padding:12px;border-radius:10px;"
            f"margin-bottom:10px;color:#f1f5f9'>"
            f"<b>{tick} ({pos})</b> — "
            f"<span style='color:#22d3ee'>${amt:,.0f}</span><br>{rat}"
        )

        # Add source link if available and valid URL
        if re.match(r'^https?://', src):
            card += f"<br><a href='{src}' target='_blank' style='color:#60a5fa;'>Source ↗</a>"

        card += "</div>"
        st.markdown(card, unsafe_allow_html=True)

