import streamlit as st
import pandas as pd
import re
from typing import List

def render_strategy_cards(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No strategies available.")
        return
    for i, row in df.iterrows():
        raw_rationale = row.rationale
        thesis_text = raw_rationale.get("thesis") if isinstance(raw_rationale, dict) else str(raw_rationale)
        first_sentence = thesis_text.split(".")[0].strip()
        headline_words = first_sentence.split()[:5]
        headline = " ".join(headline_words) + "…"
        chosen = st.session_state.get("chosen_strategy") or {}
        selected = chosen.get("name") == row.name
        box_color = "#10b981" if selected else "#334155"
        with st.container():
            st.markdown(
                f"""
                <div style="border: 1px solid {box_color}; border-radius: 10px; padding: 16px; margin-bottom: 16px; background-color: #1e1f24;">
                <div style="display:flex; justify-content: space-between; align-items:center;">
                    <div style="font-size: 18px; font-weight: 600;">{headline}</div>
                    <div style="font-size: 13px; background-color: #334155; color: #f8fafc; padding: 4px 10px; border-radius: 6px;">
                        Variant {row.variant}
                    </div>
                </div>
                <div style="margin-top: 8px;">
                    <b>Risk Reduction:</b> {row.risk_reduction_pct}%   
                    <b>Cost:</b> {row.get("aggregate_cost_pct", 0):.1f}% of capital   
                    <b>Horizon:</b> {row.get("horizon_months", "—")} months
                </div>
                <details style="margin-top: 12px; color: #e2e8f0;">
                    <summary style="cursor: pointer;">📖 View Rationale & Trade-offs</summary>
                    <div style="margin-top: 8px; line-height: 1.6;">
                        {(
                            f"• {raw_rationale.get('thesis', '').rstrip('.')}.<br>• {raw_rationale.get('tradeoff', '').rstrip('.')}"
                            if isinstance(raw_rationale, dict)
                            else "<br>".join(f"• {s.strip()}." for s in str(raw_rationale).split(".") if s.strip())
                        )}
                    </div>
                    <form method="post">
                        <button type="submit" style="margin-top: 12px; padding: 6px 12px; background-color: #10b981; color: white; border: none; border-radius: 6px; cursor: pointer;" name="select_strategy_{i}">✔️ Select this strategy</button>
                    </form>
                </details>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.session_state.get(f"select_strategy_{i}"):
                st.session_state.chosen_strategy = row.to_dict()

def clean_md(md: str) -> str:
    md = re.sub(r"(\d)(?=[A-Za-z])", r"\1 ", md)
    return md.replace("*", "").replace("_", "")

def render_rationale(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No hedge rationale to show.")
        return
    total = df["Amount ($)"].sum()
    st.markdown(
        f"A total of **${total:,.0f}** was allocated to hedge instruments to mitigate key risks in the portfolio.\n\nBelow is the explanation for each hedge component:"
    )
    for _, row in df.iterrows():
        tick = row.get("Ticker", "—").strip()
        pos = row.get("Position", "—").title()
        amt = row.get("Amount ($)", 0)
        rat = row.get("Rationale", "No rationale provided").strip()
        src = row.get("Source", "").strip()
        card = (
            f"<div style='background:#1e293b;padding:12px;border-radius:10px;margin-bottom:10px;color:#f1f5f9'>"
            f"<b>{tick} ({pos})</b> — <span style='color:#22d3ee'>${amt:,.0f}</span><br>{rat}"
        )
        if re.match(r'^https?://', src):
            card += f"<br><a href='{src}' target='_blank' style='color:#60a5fa;'>Source ↗</a>"
        card += "</div>"
        st.markdown(card, unsafe_allow_html=True)

def render_backtest_chart(unhedged_values: List[float], hedged_values: List[float], dates: List[str]):
    """
    Render a line chart comparing unhedged vs. hedged portfolio values.
    Args:
        unhedged_values: List of unhedged portfolio values over time.
        hedged_values: List of hedged portfolio values over time.
        dates: List of corresponding dates.
    """
    df = pd.DataFrame({
        'Date': pd.to_datetime(dates),
        'Unhedged Portfolio': unhedged_values,
        'Hedged Portfolio': hedged_values
    })
    st.line_chart(df.set_index('Date'))