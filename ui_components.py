import streamlit as st
import pandas as pd
import re
from typing import List

# ---------- 1. Global styles ----------
st.markdown(
    """
    <style>
    .stExpander > div,
    .card {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.10);
    }
    .card {
        background-color: #1E1F24;
        padding: 20px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- 2. Helpers ----------
def clean_md(md: str) -> str:
    """Strip markdown artefacts that confuse Streamlit."""
    md = re.sub(r"(\d)(?=[A-Za-z])", r"\1 ", md)
    return md.replace("*", "").replace("_", "")

def render_backtest_chart(
    unhedged_values: List[float],
    hedged_values: List[float],
    dates: List[str],
):
    """Simple line‑chart helper used by app.py."""
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "Unhedged Portfolio": unhedged_values,
            "Hedged Portfolio": hedged_values,
        }
    )
    st.line_chart(df.set_index("Date"))

# ---------- 3. Main card renderer ----------
def render_strategy_cards(df: pd.DataFrame) -> None:
    """
    Show each strategy as a card with a **real** Streamlit button so the
    selection propagates through st.session_state (HTML forms don’t).
    """
    if df.empty:
        st.info("No strategies generated yet.")
        return

    for i, row in df.iterrows():
        raw_rationale = row.rationale
        thesis = (
            raw_rationale.get("thesis")
            if isinstance(raw_rationale, dict)
            else str(raw_rationale)
        )
        headline = " ".join(thesis.split()[:5]) + "…"

        chosen = st.session_state.get("chosen_strategy") or {}
        selected = chosen.get("name") == row.name
        border = "#10b981" if selected else "#60A5FA"

        with st.container():
            st.markdown(
                f"""
                <div class="card" style="border: 1px solid {border};">
                  <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-size:20px; font-weight:600;">{headline}</div>
                    <div style="font-size:14px; background:#334155; color:#F8FAFC; padding:4px 10px; border-radius:6px;">
                      Variant {row.variant}
                    </div>
                  </div>
                  <div style="margin-top:8px; line-height:1.8;">
                    <b>Risk Reduction:</b> {row.risk_reduction_pct}%  
                    <b>Cost:</b> {row.get('aggregate_cost_pct',0):.1f}%  
                    <b>Horizon:</b> {row.get('horizon_months','—')} mo
                  </div>
                  <details style="margin-top:12px; color:#E2E8F0;">
                    <summary style="cursor:pointer;">📖 View Rationale & Trade‑offs</summary>
                    <div style="margin-top:8px; line-height:1.6;">
                      {(
                        f"• {raw_rationale.get('thesis','').rstrip('.')}.<br>• {raw_rationale.get('tradeoff','').rstrip('.')}"
                        if isinstance(raw_rationale, dict)
                        else "<br>".join(f"• {s.strip()}." for s in str(raw_rationale).split('.') if s.strip())
                      )}
                    </div>
                  </details>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                "✔️ Select this strategy",
                key=f"select_strategy_{i}",
                help="Activate this strategy and unlock back‑testing",
            ):
                st.session_state.chosen_strategy = row.to_dict()
                st.session_state.strategy_df = df         # keep for back‑test
                st.rerun()

# ---------- 4. Optional rationale renderer ----------
def render_rationale(df: pd.DataFrame) -> None:
    """Pretty‑print hedge rationale cards."""
    if df.empty:
        st.info("No hedge rationale to show.")
        return
    total = df["Amount ($)"].sum()
    st.markdown(
        f"A total of **${total:,.0f}** was allocated to hedges. Below is the rationale for each leg:"
    )
    for _, row in df.iterrows():
        tick = row.get("Ticker", "—").strip()
        pos = row.get("Position", "—").title()
        amt = row.get("Amount ($)", 0)
        rat = row.get("Rationale", "No rationale provided").strip()
        src = row.get("Source", "").strip()
        card = (
            f"<div class='card' style='color:#F1F5F9'>"
            f"<b>{tick} ({pos})</b> — <span style='color:#22D3EE'>${amt:,.0f}</span><br>{rat}"
        )
        if re.match(r"^https?://", src):
            card += f"<br><a href='{src}' target='_blank' style='color:#60A5FA;'>Source ↗</a>"
        card += "</div>"
        st.markdown(card, unsafe_allow_html=True)
