import streamlit as st
import pandas as pd
import re
from openai_client import ask_openai
from config import DEFAULT_MODEL

def render_strategy_cards(df: pd.DataFrame) -> None:
    """
    Render hedge strategies in a horizontal scrollable row using real Streamlit buttons.
    Hovering over the name shows only one action: 'Select Strategy'.
    """
    if df.empty:
        st.info("No strategies generated yet.")
        return

    st.markdown("<div style='display:flex; overflow-x:auto; gap:1rem; padding-bottom:1rem;'>", unsafe_allow_html=True)
    
    for i, row in df.iterrows():
        short_title = " ".join(str(row.rationale.get("thesis", row.rationale)).split()[:5]) + "…"
        border_color = "#10b981" if (
            st.session_state.get("chosen_strategy", {}).get("name") == row.name
        ) else "#60A5FA"

        button_html = f"""
            <div style='
                min-width: 200px;
                max-width: 200px;
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid {border_color};
                padding: 10px;
                border-radius: 8px;
                text-align: center;
                font-weight: 600;
                font-size: 13px;
                cursor: pointer;
            '>
                {short_title}
            </div>
        """

        if st.button(label=short_title, key=f"strategy_button_{i}"):
            st.session_state["selected_strategy_idx"] = i
            st.session_state["chosen_strategy"] = row.to_dict()
            st.rerun()

        st.markdown(button_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

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

