from __future__ import annotations
import streamlit as st
import pandas as pd
from io import StringIO
from ui_components import render_strategy_cards, render_rationale, clean_md
from utils import search_tickers, fetch_prices, web_risk_scan, fetch_backtest_data
from strategy_generator import generate_strategies
from backtest import backtest_strategy
from config import DEFAULT_MODEL
from pathlib import Path
from openai_client import ask_openai

# Set Streamlit page configuration to wide mode
st.set_page_config(page_title="Hedge Strategy Chatbot", layout="wide")

# Custom CSS for professional styling
st.markdown(f"<style>{Path('style.css').read_text()}</style>", unsafe_allow_html=True)

# Session State Initialization
defaults = {
    "history": [],
    "portfolio": [],
    "portfolio_alloc": {},
    "alloc_df": None,
    "outlook_md": None,
    "risk_cache": {},
    "risk_ignore": [],
    "chosen_strategy": None,
    "strategy_legs": {},
    "strategy_history": [],
    "backtest_duration": 12  # months
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Sidebar: Investor Profile
with st.sidebar.expander("📌 Investor Profile", expanded=False):
    st.selectbox(
        label="Investor experience",
        options=["Beginner", "Intermediate", "Expert"],
        index=["Beginner", "Intermediate", "Expert"].index(st.session_state.get("experience_level", "Expert")),
        format_func=lambda x: f"Experience: {x}",
        key="experience_level"
    )
    st.selectbox(
        label="Explanation detail preference",
        options=["Just the strategy", "Explain the reasoning", "Both"],
        index=["Just the strategy", "Explain the reasoning", "Both"].index(st.session_state.get("explanation_pref", "Just the strategy")),
        format_func=lambda x: f"Detail level: {x}",
        key="explanation_pref"
    )
    st.slider(
        label="Time horizon (months):",
        min_value=1,
        max_value=24,
        value=st.session_state.get("time_horizon", 6),
        key="time_horizon"
    )
    # Instruments
    exp_map = {
        "Beginner": ["Inverse ETFs", "Commodities"],
        "Intermediate": ["Put Options", "Inverse ETFs", "Commodities"],
        "Expert": ["Put Options", "Collar Strategy", "Inverse ETFs", "Short Selling", "Volatility Hedges", "Commodities", "FX Hedges"]
    }
    all_instruments = list({instr for v in exp_map.values() for instr in v})
    current = st.session_state.experience_level
    if st.session_state.get("prev_exp") != current:
        st.session_state.allowed_instruments = exp_map[current]
        st.session_state.prev_exp = current
    st.multiselect(
        "Allowed hedge instruments:",
        options=all_instruments,
        default=st.session_state.allowed_instruments,
        key="allowed_instruments"
    )

# Sidebar: Session Tools
with st.sidebar.expander("🧹 Session Tools", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        suggest_clicked = st.button("⚡\nGenerate", key="suggest_btn", help="Generate new hedge strategies")
    with col2:
        show_history_clicked = st.button("📈\nHistory", key="show_history_btn", help="View strategy history")
    with col3:
        clear_portfolio_clicked = st.button("🗂️\nReset", key="clear_portfolio_btn", help="Clear current portfolio")
    with col4:
        clear_chat_clicked = st.button("💬\nClear", key="clear_chat_btn", help="Clear chat history")
    
    # Handle button clicks
    if show_history_clicked:
        if st.session_state.strategy_history:
            st.subheader("📊 Strategy History")
            for idx, run in enumerate(reversed(st.session_state.strategy_history), start=1):
                with st.expander(f"Run {idx} — {run['timestamp']} | Horizon: {run['horizon']} mo", expanded=False):
                    st.markdown(f"**Capital**: ${run['capital']:,.0f}  \n**Beta Band**: {run['beta_band'][0]}–{run['beta_band'][1]}")
                    st.dataframe(run["strategy_df"], use_container_width=True)
                    st.markdown("**Rationale**")
                    st.markdown(run["rationale_md"])
        else:
            st.info("No strategy history available yet.")
    
    if clear_portfolio_clicked:
        st.session_state.portfolio = []
        st.session_state.portfolio_alloc = {}
        st.session_state.alloc_df = None
        st.rerun()
    
    if clear_chat_clicked:
        st.session_state.history = []
        st.rerun()
    
    # Additional controls
    st.slider(
        label="Backtest period (months):",
        min_value=1,
        max_value=24,
        value=st.session_state.backtest_duration,
        key="backtest_duration"
    )
    
    if st.button("🗑️ Clear Strategy History", key="clear_strategy_btn"):
        st.session_state.strategy_history = []
        st.rerun()

# Main UI
st.title("Equity Strategy Assistant")

# Portfolio Uploader
st.subheader("📊 Portfolio")
with st.expander("Upload Portfolio (CSV)", expanded=True):
    up = st.file_uploader("Upload your portfolio", type=["csv"])
    if up:
        try:
            raw = up.getvalue().decode('utf-8')
            df = pd.read_csv(StringIO(raw), engine='python', on_bad_lines='warn')
        except Exception as e:
            st.error("Error parsing CSV.")
            st.exception(e)
            st.stop()

        if not set(["Ticker","Amount ($)"]).issubset(df.columns):
            st.error("CSV needs 'Ticker' and 'Amount ($)' columns.")
            st.stop()
        df['Ticker']=df['Ticker'].astype(str).str.upper()
        if 'Stop-Loss ($)' not in df.columns:
            df['Stop-Loss ($)']=None

        # save
        st.session_state.alloc_df = df[['Ticker','Amount ($)','Stop-Loss ($)']]
        st.session_state.portfolio = df['Ticker'].tolist()
        st.session_state.portfolio_alloc = dict(zip(df['Ticker'], df['Amount ($)']))

        # display
        prices = fetch_prices(st.session_state.portfolio, period="2d")
        disp = st.session_state.alloc_df.copy()
        if not prices.empty:
            last,prev=prices.iloc[-1],prices.iloc[-2]
            disp['Price']=last.reindex(st.session_state.portfolio).round(2).values
            disp['Δ 1d %']=((last-prev)/prev*100).reindex(st.session_state.portfolio).round(2).values
        else:
            disp['Price']=0.0;disp['Δ 1d %']=0.0
        st.dataframe(disp, use_container_width=True)
    else:
        st.warning("Please upload a portfolio CSV to continue.")
        st.stop()

# Strategy Generation
if suggest_clicked:
    if not st.session_state.portfolio_alloc:
        st.warning("No portfolio — cannot suggest strategies.")
        st.stop()

    total = sum(st.session_state.portfolio_alloc.values())
    risks = "; ".join([ risk for ticker in st.session_state.portfolio for risk, _ in st.session_state.risk_cache.get(ticker, [])])
    strat_df = generate_strategies(
        model=DEFAULT_MODEL,
        portfolio=st.session_state.portfolio,
        total_capital=total,
        horizon=st.session_state.time_horizon,
        risk_string=risks,
        allowed_instruments=st.session_state.allowed_instruments,
        explanation_pref=st.session_state.explanation_pref,
        experience_level=st.session_state.experience_level
    )
    st.session_state.strategy_df = strat_df
    st.subheader("🛡️ Recommended Hedging Strategies")
    render_strategy_cards(strat_df)

# Backtesting
if st.session_state.chosen_strategy:
    st.info(f"Chosen: {st.session_state.chosen_strategy['name']}")
    if st.button("📊 Run Backtest"):
        port = st.session_state.portfolio
        hedge = [leg['instrument'].split()[0] for leg in st.session_state.strategy_legs.get(st.session_state.strategy_df.index[st.session_state.strategy_df['name'] == st.session_state.chosen_strategy['name']][0], [])]
        data = fetch_backtest_data(port + hedge, period=f"{st.session_state.backtest_duration}m")
        result = backtest_strategy(
            st.session_state.chosen_strategy,
            {t: data[t] for t in port if t in data},
            {h: data[h] for h in hedge if h in data},
            total
        )
        st.subheader("Backtest Results")
        st.markdown(f"""
        - **Unhedged**: ${result['unhedged_final_value']:,.2f}
        - **Hedged**: ${result['hedged_final_value']:,.2f}
        - **Risk Reduction**: {result['risk_reduction_pct']:.2f}%
        - **Max Drawdown (Unhedged)**: {result['max_drawdown_unhedged']:.2f}%
        - **Max Drawdown (Hedged)**: {result['max_drawdown_hedged']:.2f}%
        - **Total Hedge Cost**: ${result['total_cost']:,.2f}
        """)
        if 'dates' in result:
            render_backtest_chart(result['unhedged_values'], result['hedged_values'], result['dates'])

# Quick Chat
st.divider()
st.markdown("### 💬 Quick chat")
for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)
if q := st.chat_input("Ask anything…"):
    st.session_state.history.append(("user", q))
    resp = ask_openai(DEFAULT_MODEL, "You are a helpful market analyst.", f"User portfolio: {','.join(st.session_state.portfolio)}\n{q}")
    st.session_state.history.append(("assistant", resp))
    st.rerun()