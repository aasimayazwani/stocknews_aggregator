import streamlit as st
import pandas as pd
from openai_client import ask_openai
from config import DEFAULT_MODEL

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
                      Variant: {row.variant}
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
                "✔️ Select this strategy",
                key=f"select_strategy_{i}",
                help="Select this strategy and get a brief explanation",
            ):
                # Set the chosen strategy
                st.session_state.chosen_strategy = row.to_dict()
                st.session_state.strategy_df = df
                
                # Prepare the prompt for OpenAI
                strategy_name = row.get('name', 'Unknown Strategy')
                variant = row.get('variant', 'Unknown Variant')
                prompt = f"Provide a brief explanation of the hedging strategy '{strategy_name}' with variant '{variant}'."
                
                # Call OpenAI to get the explanation
                explanation = ask_openai(
                    model=DEFAULT_MODEL,
                    system_prompt="You are a financial expert providing brief explanations of hedging strategies.",
                    user_prompt=prompt
                )
                
                # Append the explanation to the chat history
                message = f"**Explanation for {strategy_name} (Variant: {variant}):**\n{explanation}"
                st.session_state.history.append(("assistant", message))
                st.rerun()

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
                message = f"**Rationale for {strategy_name} (Variant: {row.variant}):**\n{rationale_text}"
                st.session_state.history.append(("assistant", message))
                st.rerun()

def clean_md(md: str) -> str:
    md = re.sub(r"(\d)(?=[A-Za-z])", r"\1 ", md)
    return md.replace("*", "").replace("_", "")

def render_rationale(df: pd.DataFrame) -> None:
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
        tick   = row.get("Ticker", "—").strip()
        pos    = row.get("Position", "—").title()
        amt    = row.get("Amount ($)", 0)
        rat    = row.get("Rationale", "No rationale provided").strip()
        src    = row.get("Source", "").strip()

        card  = (
            f"<div style='background:#1e293b;padding:12px;border-radius:10px;"
            f"margin-bottom:10px;color:#f1f5f9'>"
            f"<b>{tick} ({pos})</b> — "
            f"<span style='color:#22d3ee'>${amt:,.0f}</span><br>{rat}"
        )

        if re.match(r'^https?://', src):
            card += f"<br><a href='{src}' target='_blank' style='color:#60a5fa;'>Source ↗</a>"

        card += "</div>"
        st.markdown(card, unsafe_allow_html=True)
