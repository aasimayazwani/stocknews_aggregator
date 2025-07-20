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

# ---------- 2. Helper functions ----------
def clean_md(md: str) -> str:
    """Remove stray characters and cram‑together digits/letters."""
    md = re.sub(r"(\d)(?=[A-Za-z])", r"\1 ", md)
    return md.replace("*", "").replace("_", "")

def render_backtest_chart(
    unhedged_values: List[float],
    hedged_values: List[float],
    dates: List[str],
):
    """Simple line chart comparing unhedged vs. hedged curves."""
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
    Render each strategy in a nice card.  Selecting a card sets
    st.session_state.chosen_strategy and triggers st.rerun().
    """
    if df.empty:
        st.info("No strategies available.")
        return

    for i, row in df.iterrows():
        # Build a concise headline from the first sentence
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
                    <b>Risk Reduction:</b> {row.risk_reduction_pct}%  
                    <b>Cost:</b> {row.get('aggregate_cost_pct',0):.1f}% of capital  
                    <b>Horizon:</b> {row.get('horizon_months','—')} months
                  </div>
                  <details style="margin-top:12px; color:#E2E8F0;">
                    <summary style="cursor:pointer;">📖 View Rationale & Trade‑offs</summary>
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

            # **Real Streamlit button ― no HTML form**
            if st.button(
                "✔️ Select this strategy",
                key=f"select_strategy_{i}",
                help="Set this strategy as active and enable back‑testing",
            ):
                st.session_state.chosen_strategy = row.to_dict()
                st.session_state.strategy_df = df  # keep for back‑test lookup
                st.rerun()
