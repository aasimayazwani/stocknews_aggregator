import streamlit as st
import pandas as pd

def render_strategy_cards(df: pd.DataFrame) -> None:
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
                    <b>Risk Reduction:</b> {row.risk_reduction_pct}%  
                    <b>Cost:</b> {row.get('aggregate_cost_pct',0):.1f}%  
                    <b>Horizon:</b> {row.get('horizon_months','—')} mo
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                "📖 View Rationale",
                key=f"view_rationale_{i}",
                help="Display the rationale in the chat interface",
            ):
                rationale_text = (
                    f"- {raw_rationale.get('thesis', '').rstrip('.')}\n- {raw_rationale.get('tradeoff', '').rstrip('.')}"
                    if isinstance(raw_rationale, dict)
                    else "\n".join(f"- {s.strip()}." for s in str(raw_rationale).split('.') if s.strip())
                )
                strategy_name = row.get('name', 'Unknown Strategy')
                message = f"**Rationale for {strategy_name} (Variant {row.variant}):**\n{rationale_text}"
                st.session_state.history.append(("assistant", message))
                st.rerun()

            if st.button(
                "✔️ Select this strategy",
                key=f"select_strategy_{i}",
                help="Activate this strategy and unlock back-testing",
            ):
                st.session_state.chosen_strategy = row.to_dict()
                st.session_state.strategy_df = df
                st.rerun()

def render_rationale(rationale: str | dict) -> None:
    st.subheader("Rationale")
    if isinstance(rationale, dict):
        st.write(f"**Thesis**: {rationale.get('thesis', 'No thesis provided.')}")
        st.write(f"**Trade-off**: {rationale.get('tradeoff', 'No trade-off provided.')}")
    else:
        st.write(rationale)

def render_backtest_chart(df: pd.DataFrame) -> None:
    st.subheader("Backtest Results")
    st.line_chart(df)