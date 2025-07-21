from __future__ import annotations
import streamlit as st
import pandas as pd
from io import StringIO
from ui_components import render_strategy_cards, render_rationale, clean_md, render_backtest_chart
from utils import search_tickers, fetch_prices, web_risk_scan, fetch_backtest_data
from strategy_generator import generate_strategies
from backtest import backtest_strategy
from config import DEFAULT_MODEL
from openai_client import ask_openai

st.set_page_config(page_title="Hedge Strategy Chatbot", layout="centered")

# ------------------------ Custom CSS ------------------------
st.markdown(
    """
    <style>
    .stExpander > div {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background-color: #1E3A8A;
        color: white;
        padding: 8px 16px;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #1E40AF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Session State Initialization ------------------
defaults = {
    "history": [],
    "portfolio": [],                 # <-- empty until CSV upload
    "portfolio_alloc": {},           # <-- empty until CSV upload
    "outlook_md": None,
    "risk_cache": {},
    "risk_ignore": [],
    "chosen_strategy": None,
    "strategy_history": [],
    "backtest_duration": 12  # months
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ------------------ Sidebar: Investor Profile ------------------
with st.sidebar.expander("📌 Investor Profile", expanded=False):
    experience = st.selectbox(
        label="Investor experience",
        options=["Beginner", "Intermediate", "Expert"],
        index=["Beginner", "Intermediate", "Expert"].index(st.session_state.get("experience_level", "Expert")),
        format_func=lambda x: f"Experience: {x}",
        key="experience_level"
    )
    detail_level = st.selectbox(
        label="Explanation detail preference",
        options=["Just the strategy", "Explain the reasoning", "Both"],
        index=["Just the strategy", "Explain the reasoning", "Both"].index(st.session_state.get("explanation_pref", "Just the strategy")),
        format_func=lambda x: f"Detail level: {x}",
        key="explanation_pref"
    )
    horizon = st.slider(
        label="Time horizon (months):",
        min_value=1,
        max_value=24,
        value=st.session_state.get("time_horizon", 6),
        key="time_horizon"
    )

    experience_defaults = {
        "Beginner": ["Inverse ETFs", "Commodities"],
        "Intermediate": ["Put Options", "Inverse ETFs", "Commodities"],
        "Expert": ["Put Options", "Collar Strategy", "Inverse ETFs", "Short Selling", "Volatility Hedges", "Commodities", "FX Hedges"]
    }
    all_options = ["Put Options", "Collar Strategy", "Inverse ETFs", "Short Selling", "Volatility Hedges", "Commodities", "FX Hedges"]
    current_exp = st.session_state.get("experience_level", "Beginner")
    if "prev_experience" not in st.session_state or st.session_state.prev_experience != current_exp:
        st.session_state.allowed_instruments = experience_defaults.get(current_exp, [])
        st.session_state.prev_experience = current_exp
    st.multiselect(
        "Allowed hedge instruments:",
        options=all_options,
        default=st.session_state.allowed_instruments,
        key="allowed_instruments"
    )

# ------------------ Sidebar: Session Tools ------------------
with st.sidebar.expander("🧹 Session Tools", expanded=False):
    with st.sidebar.expander("🧠 Previous Strategies", expanded=True):
        history = st.session_state.get("strategy_history", [])
        if not history:
            st.info("No previous strategies yet.")
        else:
            for idx, run in reversed(list(enumerate(history))):
                with st.expander(f"Run {idx+1} — {run['timestamp']} | Horizon: {run['horizon']} mo"):
                    st.markdown(f"**Capital**: ${run['capital']:,.0f}  \n**Beta Band**: {run['beta_band'][0]}–{run['beta_band'][1]}")
                    st.dataframe(run["strategy_df"], use_container_width=True)
                    st.markdown("**Strategy Rationale**")
                    st.markdown(run["rationale_md"])

    suggest_clicked = st.sidebar.button("🚀 Suggest strategy", type="primary", use_container_width=True)

    with st.sidebar.expander("⏳ Backtest Duration", expanded=True):
        st.session_state.backtest_duration = st.slider(
            label="Backtest period (months):",
            min_value=1,
            max_value=24,
            value=st.session_state.backtest_duration,
            key="backtest_duration"
        )

    if st.button("🗑️ Clear Portfolio"):
        st.session_state.portfolio = []
        st.session_state.portfolio_alloc = {}
    if st.button("🧽 Clear Chat History"):
        st.session_state.chat_history = []
    if st.button("🗑️ Clear Strategy History"):
        st.session_state.strategy_history = []

# ------------------ Main UI ------------------
st.title("Equity Strategy Assistant")

# ------------------ Portfolio UI ------------------
st.subheader("📊 Portfolio")
with st.expander("Upload & View Portfolio", expanded=True):
    uploaded_file = st.file_uploader("Upload your portfolio (CSV)", type=["csv"])
    if uploaded_file:
        try:
            content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            df = pd.read_csv(StringIO(content), engine="python", on_bad_lines="warn")
        except Exception as e:
            st.error("❌ Error reading CSV. Please check for missing quotes, commas, or formatting issues.")
            st.exception(e)
            st.stop()

        required_cols = ["Ticker", "Amount ($)"]
        if not all(col in df.columns for col in required_cols):
            st.error("CSV must contain at least 'Ticker' and 'Amount ($)' columns.")
            st.stop()

        if "Stop-Loss ($)" not in df.columns:
            df["Stop-Loss ($)"] = None

        df["Ticker"] = df["Ticker"].astype(str).str.upper()

        # Save into session state
        st.session_state.alloc_df = df[["Ticker", "Amount ($)", "Stop-Loss ($)"]]
        st.session_state.portfolio = df["Ticker"].tolist()
        st.session_state.portfolio_alloc = dict(zip(df["Ticker"], df["Amount ($)"]))

        # Display enriched portfolio table
        tickers = st.session_state.portfolio
        prices_df = fetch_prices(tickers, period="2d")
        display_df = st.session_state.alloc_df.copy()

        if not prices_df.empty:
            last = prices_df.iloc[-1]
            prev = prices_df.iloc[-2]
            display_df["Price"] = last.reindex(tickers).round(2).values
            display_df["Δ 1d %"] = ((last - prev) / prev * 100).reindex(tickers).round(2).values
        else:
            display_df["Price"] = 0.0
            display_df["Δ 1d %"] = 0.0

        st.dataframe(display_df, use_container_width=True)
    else:
        st.warning("Please upload a valid portfolio CSV to proceed.")
        st.stop()

# ------------------ Strategy Suggestion ------------------
if suggest_clicked:
    total_capital = sum(st.session_state.portfolio_alloc.values())
    risk_string = "; ".join(title for t in st.session_state.portfolio for title, _url in st.session_state.risk_cache.get(t, []))

    df_strat = generate_strategies(
        model=DEFAULT_MODEL,
        portfolio=st.session_state.portfolio,
        total_capital=total_capital,
        horizon=st.session_state.time_horizon,
        risk_string=risk_string,
        allowed_instruments=st.session_state.allowed_instruments,
        explanation_pref=st.session_state.explanation_pref,
        experience_level=st.session_state.experience_level
    )
    st.session_state.strategy_df = df_strat
    st.subheader("🛡️ Recommended Hedging Strategies")
    render_strategy_cards(df_strat)

    # Backtesting block
    if st.session_state.chosen_strategy:
        st.info(f"**Chosen strategy:** {st.session_state.chosen_strategy['name']}")
        if st.button("📊 Run Backtest"):
            portfolio_tickers = st.session_state.portfolio
            hedge_tickers = list(
                set(
                    leg['instrument'].split()[0]
                    for leg in st.session_state.strategy_legs.get(
                        df_strat.index[df_strat['name'] == st.session_state.chosen_strategy['name']].tolist()[0], []
                    )
                )
            )
            all_tickers = portfolio_tickers + hedge_tickers
            backtest_data = fetch_backtest_data(all_tickers, period=f"{st.session_state.backtest_duration}m")

            backtest_results = backtest_strategy(
                st.session_state.chosen_strategy,
                {t: backtest_data[t] for t in portfolio_tickers if t in backtest_data},
                {t: backtest_data[t] for t in hedge_tickers if t in backtest_data},
                total_capital
            )

            st.subheader("Backtest Results")
            st.markdown(f"""
            - **Unhedged Final Value**: ${backtest_results['unhedged_final_value']:,.2f}
            - **Hedged Final Value**: ${backtest_results['hedged_final_value']:,.2f}
            - **Risk Reduction**: {backtest_results['risk_reduction_pct']:.2f}%
            - **Max Drawdown (Unhedged)**: {backtest_results['max_drawdown_unhedged']:.2f}%
            - **Max Drawdown (Hedged)**: {backtest_results['max_drawdown_hedged']:.2f}%
            - **Total Hedge Cost**: ${backtest_results['total_cost']:,.2f}
            """)

            if backtest_results.get('dates'):
                st.subheader("Portfolio Value Over Time")
                render_backtest_chart(
                    backtest_results['unhedged_values'],
                    backtest_results['hedged_values'],
                    backtest_results['dates']
                )

# ------------------ Quick Chat ------------------
st.divider()
st.markdown("### 💬 Quick chat")

for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

if q := st.chat_input("Ask anything…"):
    ctx = f"User portfolio: {', '.join(st.session_state.portfolio)}. Focus: All stocks."
    st.session_state.history.append(("user", q))
    ans = ask_openai(DEFAULT_MODEL, "You are a helpful market analyst.", ctx + "\n\n" + q)
    st.session_state.history.append(("assistant", ans))
    st.rerun()
