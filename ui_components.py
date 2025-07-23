import streamlit as st
import pandas as pd
import re
from openai_client import ask_openai
from config import DEFAULT_MODEL

def render_strategy_cards(df: pd.DataFrame) -> None:
    """
    Render a condensed 1-row or 2-row layout of hedge strategies with only one hover action over the strategy name.
    Hovering over the name will display a button to select the strategy.
    """
    if df.empty:
        st.info("No strategies generated yet.")
        return

    # Set number of rows and columns
    num_columns = min(len(df), 4)  # adjust depending on screen size/responsiveness
    columns = st.columns(num_columns)

    for i, row in df.iterrows():
        col = columns[i % num_columns]
        with col:
            chosen = st.session_state.get("chosen_strategy") or {}
            selected = chosen.get("name") == row.name
            border = "#10b981" if selected else "#60A5FA"

            headline = str(row.rationale.get("thesis", row.rationale))
            short_title = " ".join(headline.split()[:5]) + "…"

            st.markdown(
                f"""
                <style>
                .strategy-name-{i} {{
                    font-weight: bold;
                    font-size: 16px;
                    color: #f8fafc;
                    background: #1e293b;
                    padding: 6px 10px;
                    border: 1px solid {border};
                    border-radius: 8px;
                    margin: 6px 0;
                    position: relative;
                    cursor: pointer;
                }}
                .strategy-name-{i}:hover .select-button-{i} {{
                    display: block;
                }}
                .select-button-{i} {{
                    display: none;
                    position: absolute;
                    top: 100%;
                    left: 0;
                    margin-top: 4px;
                    background: #1e40af;
                    color: white;
                    padding: 4px 12px;
                    font-size: 12px;
                    border-radius: 6px;
                }}
                </style>
                <div class="strategy-name-{i}" onclick="window.location.search = '?selected_strategy_idx={i}';">
                    {short_title}
                    <div class="select-button-{i}">Select Strategy</div>
                </div>
                """,
                unsafe_allow_html=True
            )


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

