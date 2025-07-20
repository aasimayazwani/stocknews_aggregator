import json
import textwrap
import streamlit as st
import pandas as pd
from openai_client import ask_openai

def generate_strategies(model, portfolio, total_capital, horizon, risk_string, allowed_instruments, explanation_pref, experience_level):
    experience_note = {
        "Beginner": "Use plain language and define jargon the first time you use it.",
        "Intermediate": "Assume working knowledge of finance; keep explanations concise.",
        "Expert": "Write in professional sell-side style; no hand-holding."
    }[experience_level]
    rationale_rule = {
        "Just the strategy": "Each *Rationale* must be **≤ 25 words (one-two sentence)**.",
        "Explain the reasoning": "Each *Rationale* must be **2 sentences totalling ≈ 30-60 words** (logic + risk linkage).",
        "Both": "Each *Rationale* must be **3 sentences totalling ≈ 60-90 words** – 1️⃣ logic, 2️⃣ quantitative context, 3️⃣ trade-offs."
    }[explanation_pref]
    SYSTEM_JSON = textwrap.dedent("""
        You are a senior equity-derivatives strategist.
        Return ONE valid JSON object:
        strategies: [
            {
                name: string,
                variant: string,
                score: float,
                risk_reduction_pct: int,
                horizon_months: int,
                legs: [
                    {
                        instrument: string,
                        position: string,
                        notional_pct: float,
                        cost_pct_capital: float,
                        expiry: string
                    }
                ],
                aggregate_cost_pct: float,
                rationale: {
                    thesis: string,
                    tradeoff: string
                }
            }
        ]
        Return JSON only.
    """).strip()
    USER_JSON = textwrap.dedent(f"""
        Portfolio tickers: {', '.join(portfolio)}
        Total capital: ${total_capital:,.0f}
        Time horizon: {horizon} months
        Headline risk exposures: {risk_string or 'none'}
        Allowed hedge instruments: {', '.join(allowed_instruments)}
        Requirements:
        • 1-3 hedge legs per strategy
        • Total premium ≤ 2.0% of capital
        • Legs may hedge at index, sector, or single-name level
        • Show expiry explicitly (e.g. Sep-24).
        Objective:
        Generate 3–4 differentiated hedge strategies that reduce downside risk exposure using liquid, cost-efficient instruments.
        {experience_note}
        {rationale_rule}
        Return JSON only.
    """).strip()
    with st.spinner("⚙️ Generating multiple hedging strategies…"):
        raw_json = ask_openai(model=model, system_prompt=SYSTEM_JSON, user_prompt=USER_JSON)
        if not raw_json.strip().startswith("{"):
            st.error("❌ LLM did not return valid JSON.")
            st.code(raw_json.strip() or "[Empty response]", language="text")
            return pd.DataFrame()
    try:
        data = json.loads(raw_json)
        df_strat = pd.DataFrame([{k: v for k, v in s.items() if k != "legs"} for s in data["strategies"]])
        st.session_state.strategy_legs = {idx: s["legs"] for idx, s in enumerate(data["strategies"])}
        return df_strat
    except (json.JSONDecodeError, KeyError) as err:
        st.error(f"❌ LLM returned invalid JSON: {err}")
        return pd.DataFrame()