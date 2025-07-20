from __future__ import annotations

import streamlit as st
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta

from ui_components import (
    render_strategy_cards,
    render_rationale,
    clean_md,
    render_backtest_chart,
)
from utils import search_tickers, fetch_prices, web_risk_scan, fetch_backtest_data
from strategy_generator import generate_strategies
from backtest import backtest_strategy
from config import DEFAULT_MODEL
from openai_client import ask_openai

# ---------- 1. Page + CSS ----------
st.set_page_config(page_title="Hedge Strategy Chatbot", layout="centered")
st.markdown(
    """
    <style>
    .stExpander > div {border:1px solid #E5E7EB;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1);}
    .stButton > button {background:#1E3A8A;color:#FFF;padding:8px 16px;border-radius:5px;}
    .stButton > button:hover {background:#1E40AF;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- 2. Session state ----------
defaults = {
    "history": [],
    "portfolio": ["AAPL", "MSFT"],
    "outlook_md": None,
    "risk_cache": {},
    "risk_ignore": [],
    "chosen_strategy": None,
    "strategy_history": [],
    "backtest_start_date": datetime.today() - timedelta(days=365),
    "backtest_end_date": datetime.today(),
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

if "portfolio_alloc" not in st.session_state:
    st.session_state.portfolio_alloc = {"AAPL": 10000, "MSFT": 10000}

# ---------- 3. Sidebar ----------
with st.sidebar.expander("\ud83d\udccc Investor Profile", expanded=False):
    experience = st.selectbox(
        "Investor experience",
        ["Beginner", "Intermediate", "Expert"],
        index=["Beginner", "Intermediate", "Expert"].index(
            st.session_state.get("experience_level", "Expert")
        ),
        key="experience_level",
    )
    detail_level = st.selectbox(
        "Explanation detail preference",
        ["Just the strategy", "Explain the reasoning", "Both"],
        index=["Just the strategy", "Explain the reasoning", "Both"].index(
            st.session_state.get("explanation_pref", "Just the strategy")
        ),
        key="explanation_pref",
    )
    horizon = st.slider(
        "Time horizon (months):",
        1,
        24,
        value=st.session_state.get("time_horizon", 6),
        key="time_horizon",
    )

# --- Session Tools ---
with st.sidebar.expander("\ud83d\udeb9 Session Tools", expanded=False):
    with st.sidebar.expander("\ud83e\udde0 Previous Strategies", expanded=False):
        history = st.session_state.get("strategy_history", [])
        if not history:
            st.info("No previous strategies yet.")
        else:
            for idx, run in reversed(list(enumerate(history))):
                with st.expander(
                    f"Run {idx+1} — {run['timestamp']} | Horizon {run['horizon']} mo"
                ):
                    st.markdown(
                        f"**Capital**: ${run['capital']:,0f}  \n**Beta Band**: {run['beta_band'][0]}–{run['beta_band'][1]}"
                    )
                    st.dataframe(run["strategy_df"], use_container_width=True)
                    st.markdown("**Strategy Rationale**")
                    st.markdown(run["rationale_md"])

    suggest_clicked = st.sidebar.button("\ud83d\ude80 Suggest strategy", type="primary")

    with st.sidebar.expander("\ud83d\uddd3\ufe0f Backtest Date Range", expanded=True):
        st.date_input("Backtest Start Date", value=st.session_state.backtest_start_date, key="backtest_start_date")
        st.date_input("Backtest End Date", value=st.session_state.backtest_end_date, key="backtest_end_date")
        if st.session_state.backtest_end_date < st.session_state.backtest_start_date:
            st.warning("End date cannot be before start date. Resetting.")
            st.session_state.backtest_end_date = st.session_state.backtest_start_date

    if st.button("\ud83d\uddd1\ufe0f Clear Portfolio"):
        st.session_state.portfolio_alloc = {}
    if st.button("\ud83e\uddfd Clear Chat History"):
        st.session_state.history = []
    if st.button("\ud83d\uddd1\ufe0f Clear Strategy History"):
        st.session_state.strategy_history = []

# ---------- 4. Main body ----------
st.title("Equity Strategy Assistant")

uploaded_file = st.file_uploader("Upload your portfolio (CSV with Ticker and Amount columns)", type=["csv"])
if uploaded_file:
    try:
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        df = pd.read_csv(StringIO(content), engine="python", on_bad_lines="warn")
        df = df.rename(columns=lambda x: x.strip())
        df["Ticker"] = df["Ticker"].astype(str).str.upper()
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
        df.dropna(subset=["Ticker", "Amount"], inplace=True)
        st.session_state.portfolio = df["Ticker"].tolist()
        st.session_state.portfolio_alloc = dict(zip(df["Ticker"], df["Amount"]))
        st.success("✅ Portfolio uploaded successfully.")
    except Exception as e:
        st.error("❌ Error reading portfolio file.")
        st.exception(e)

# ---------- 5. Strategy Suggestion ----------
if suggest_clicked:
    df_strat = generate_strategies(
        model=DEFAULT_MODEL,
        portfolio=st.session_state.portfolio,
        total_capital=sum(st.session_state.portfolio_alloc.values()),
        horizon=st.session_state.time_horizon,
        risk_string="; ".join(
            t for p in st.session_state.portfolio for t, _ in st.session_state.risk_cache.get(p, [])
        ),
        allowed_instruments=st.session_state.allowed_instruments,
        explanation_pref=st.session_state.explanation_pref,
        experience_level=st.session_state.experience_level,
    )
    st.session_state.strategy_df = df_strat
    st.subheader("\ud83d\udee1\ufe0f Recommended Hedging Strategies")
    render_strategy_cards(df_strat)

# ---------- 6. Back-test section ----------
if st.session_state.chosen_strategy:
    st.info(f"**Chosen strategy:** {st.session_state.chosen_strategy['name']}")
    if st.button("\ud83d\udcca Run Backtest", key="run_backtest_button"):
        st.session_state.run_backtest = True

    if st.session_state.get("run_backtest"):
        df_strat = st.session_state.strategy_df
        strat_idx = df_strat.index[
            df_strat["name"] == st.session_state.chosen_strategy["name"]
        ].tolist()[0]
        hedge_tickers = list(
            {
                leg["instrument"].split()[0]
                for leg in st.session_state.strategy_legs.get(strat_idx, [])
            }
        )

        tickers = st.session_state.portfolio + hedge_tickers
        backtest_data = fetch_backtest_data(
            tickers,
            start_date=st.session_state.backtest_start_date.strftime("%Y-%m-%d"),
            end_date=st.session_state.backtest_end_date.strftime("%Y-%m-%d"),
        )

        results = backtest_strategy(
            st.session_state.chosen_strategy,
            {t: backtest_data[t] for t in st.session_state.portfolio if t in backtest_data},
            {t: backtest_data[t] for t in hedge_tickers if t in backtest_data},
            sum(st.session_state.portfolio_alloc.values()),
        )

        st.subheader("Backtest Results")
        st.markdown(
            f"""
- **Unhedged Final Value:** ${results['unhedged_final_value']:,.2f}  
- **Hedged Final Value:** ${results['hedged_final_value']:,.2f}  
- **Risk Reduction:** {results['risk_reduction_pct']:.2f}%  
- **Max Drawdown (Unhedged):** {results['max_drawdown_unhedged']:.2f}%  
- **Max Drawdown (Hedged):** {results['max_drawdown_hedged']:.2f}%  
- **Total Hedge Cost:** ${results['total_cost']:,.2f}
"""
        )

        if results["dates"]:
            st.subheader("Portfolio Value Over Time")
            render_backtest_chart(
                results["unhedged_values"],
                results["hedged_values"],
                results["dates"],
            )

# ---------- 7. Chat interface ----------
st.divider()
st.markdown("### \ud83d\udcac Quick chat")

for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

if q := st.chat_input("Ask anything…"):
    ctx = f"User portfolio: {', '.join(st.session_state.portfolio)}. Focus: All stocks."
    st.session_state.history.append(("user", q))
    ans = ask_openai(DEFAULT_MODEL, "You are a helpful market analyst.", ctx + "\n\n" + q)
    st.session_state.history.append(("assistant", ans))
    st.rerun()
