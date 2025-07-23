import streamlit as st
import pandas as pd
import re
from openai_client import ask_openai
from config import DEFAULT_MODEL

def render_strategy_cards(df: pd.DataFrame) -> None:
    """
    Renders a grid of strategy cards with hover-activated details.
    Each card displays key strategy metrics and, on hover, reveals a "View Details" overlay.
    Clicking the card sets a URL parameter to trigger detailed explanation and rationale
    in the main application logic.
    """
    if df.empty:
        st.info("No strategies generated yet.")
        return

    # Use Streamlit columns for a horizontal layout (e.g., 2 columns)
    # Adjust num_columns as needed for desired layout (e.g., 1, 2, 3)
    num_columns = 2
    columns = st.columns(num_columns)

    for i, row in df.iterrows():
        raw_rationale = row.rationale
        thesis = (
            raw_rationale.get("thesis")
            if isinstance(raw_rationale, dict)
            else str(raw_rationale)
        )
        # Create a short headline from the thesis for display
        headline = " ".join(thesis.split()[:5]) + "…"

        chosen = st.session_state.get("chosen_strategy") or {}
        # Determine border color based on whether the strategy is currently selected
        selected = chosen.get("name") == row.name
        border = "#10b981" if selected else "#60A5FA"

        # Place each card in its respective column, cycling through columns
        with columns[i % num_columns]:
            # Custom HTML and CSS for the strategy card with hover effect
            # The 'onclick' event modifies the URL query parameters, which Streamlit detects
            # and triggers a rerun, allowing Python logic to react.
            st.markdown(
                f"""
                <style>
                    /* Basic styling for the strategy card */
                    .strategy-card {{
                        border: 1px solid {border};
                        padding: 1em;
                        margin-bottom: 1em;
                        border-radius: 10px; /* Rounded corners for the card */
                        cursor: pointer; /* Indicate interactivity */
                        position: relative;
                        overflow: hidden; /* Ensures the hover overlay stays within bounds */
                        transition: all 0.2s ease-in-out; /* Smooth transition for hover effects */
                        background-color: #1a202c; /* Dark background for the card */
                        color: #f1f5f9; /* Light text color */
                    }}
                    /* Hover effect: slight lift and shadow */
                    .strategy-card:hover {{
                        transform: translateY(-5px);
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                    }}
                    /* Styling for the hidden overlay that appears on hover */
                    .strategy-card .hover-overlay {{
                        position: absolute;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        background: rgba(0, 0, 0, 0.7); /* Semi-transparent dark overlay */
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        opacity: 0; /* Initially hidden */
                        transition: opacity 0.3s ease; /* Smooth fade-in/out */
                        border-radius: 10px; /* Match card border-radius */
                    }}
                    /* Make overlay visible on card hover */
                    .strategy-card:hover .hover-overlay {{
                        opacity: 1;
                    }}
                    /* Styling for the text inside the hover overlay */
                    .hover-text {{
                        color: white;
                        font-size: 1.2em;
                        font-weight: bold;
                        text-align: center;
                    }}
                    /* General text styling for card content */
                    .strategy-card b {{
                        color: #cbd5e1; /* Slightly lighter bold text */
                    }}
                </style>
                <div class="strategy-card" onclick="
                    // JavaScript to set a query parameter in the URL
                    // Streamlit will detect this change and rerun the app,
                    // allowing Python code to process the selection.
                    const url = new URL(window.location);
                    url.searchParams.set('selected_strategy_idx', {i});
                    window.location.href = url.toString();
                ">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div style="font-size:20px; font-weight:600; color:#f1f5f9;">{headline}</div>
                        <div style="font-size:14px; background:#334155; color:#F8FAFC; padding:4px 10px; border-radius:6px;">
                            Variant: {row.variant}
                        </div>
                    </div>
                    <div style="margin-top:8px; line-height:1.8; color:#cbd5e1;">
                        <b>Risk Reduction:</b> {row.risk_reduction_pct}%
                        <b>Cost:</b> {row.get('aggregate_cost_pct',0):.1f}%
                        <b>Horizon:</b> {row.get('horizon_months','—')} mo
                    </div>
                    <div class="hover-overlay">
                        <span class="hover-text">View Details</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
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

