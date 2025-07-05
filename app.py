# app.py – Market-Movement Chatbot (portfolio-aware strategy edition)
"""Streamlit application that helps investors design equity strategies,
peek at quarterly outlooks, compare prices, and chat about markets."""

from __future__ import annotations

import math, re, textwrap
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

from config import DEFAULT_MODEL          # local module
from openai_client import ask_openai      # wrapper around OpenAI API
from stock_utils import get_stock_summary # your own helper
# ───────────────────────────── UI / THEME ─────────────────────────────
st.set_page_config(page_title="Strategy Chatbot", layout="wide")

st.markdown(
    """
    <style>
    .card{background:#1e1f24;padding:18px;border-radius:12px;margin-bottom:18px;}
    .metric-tile{background:#f1f5f90D;border:1px solid #33415550;padding:18px 22px;border-radius:12px;
                 transition:background .2s;cursor:pointer;}
    .metric-tile:hover{background:#33415522;}
    .metric-title{font-weight:600;font-size:18px;margin-bottom:6px;}
    .metric-value{font-size:22px;font-weight:700;}
    .chevron{float:right;font-size:20px;line-height:18px;transform:translateY(2px);}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🎯 Equity Strategy Assistant")

# ───────────────────────────── Session State ──────────────────────────
if "history" not in st.session_state:
    st.session_state.history: List = []
if "portfolio" not in st.session_state:
    # default seed portfolio
    st.session_state.portfolio: List[str] = ["AAPL", "MSFT"]
if "outlook_md" not in st.session_state:
    st.session_state.outlook_md: str | None = None
if "last_summary_ticker" not in st.session_state:
    st.session_state.last_summary_ticker = None

def add_to_history(role: str, txt: str) -> None:
    st.session_state.history.append((role, txt))

# ───────────────────────────── Helpers ────────────────────────────────
def clean_llm_markdown(md: str) -> str:
    """Minor cleanup so model output renders nicely in Streamlit."""
    md = re.sub(r"(\d)(?=[a-zA-Z])", r"\1 ", md)
    md = re.sub(r"([a-zA-Z])(?=\d)", r"\1 ", md)
    return md.replace("*", "").replace("_", "")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_df(tickers: List[str], period: str) -> pd.DataFrame:
    df = yf.download(tickers, period=period, progress=False)["Close"]
    return df.dropna(axis=1, how="all")

# ───────────────────────── Sidebar – Settings ─────────────────────────
with st.sidebar.expander("⚙️ Settings", expanded=False):
    model = st.selectbox("OpenAI model", [DEFAULT_MODEL, "gpt-4.1-mini", "gpt-4o-mini"], 0)
    if st.button("🧹 Clear chat history"):
        st.session_state.history = []
    if st.button("🛑 Clear portfolio"):
        st.session_state.portfolio = []

show_charts = st.sidebar.checkbox("📈 Show price charts", value=False)

# ──────────────────────── Portfolio Input (Main) ──────────────────────
st.markdown("### 📌 Your Portfolio")
portfolio_input = st.text_input(
    "Enter the tickers you currently hold (comma-separated)",
    value=", ".join(st.session_state.portfolio),
)
portfolio = [t.strip().upper() for t in portfolio_input.split(",") if t.strip()]
st.session_state.portfolio = portfolio  # keep in sync

if not portfolio:
    st.info("Please enter at least one ticker to continue.")
    st.stop()

primary = st.selectbox("📍 Which stock do you want to build a strategy around?",
                       options=portfolio,
                       index=0)
other_holdings = [t for t in portfolio if t != primary]
basket = [primary] + other_holdings  # passed to the LLM later

# ─────────────────────── Sidebar – Quarterly Outlook ──────────────────
with st.sidebar.expander("🔮 Quarterly Outlook", expanded=False):
    if st.button("↻ Generate outlook", key="btn_outlook"):
        st.session_state.outlook_md = None  # reset

    if st.session_state.outlook_md is None:
        st.session_state.outlook_md = "Generating…"
        st.rerun()

    elif st.session_state.outlook_md == "Generating…":
        outlook_prompt = (
            f"Provide numeric forecasts for **EPS** and **Total Revenue** for {primary}'s "
            f"next quarter. Include your prediction, Street consensus, and beat probability "
            f"in %. Add one sentence of reasoning ending with 'Source: …'. "
            f"Return in markdown: a table plus bullets, no code fences."
        )
        with st.spinner("Contacting LLM…"):
            raw_md = ask_openai(
                model=model,
                system_prompt="You are a senior equity analyst, precise and data-driven.",
                user_prompt=outlook_prompt,
            )
        st.session_state.outlook_md = clean_llm_markdown(raw_md)
        st.rerun()

    else:
        st.markdown(f"<div class='card'>{st.session_state.outlook_md}</div>", unsafe_allow_html=True)

# ───────────────────── Snapshot (sidebar) & Summary ───────────────────
try:
    summary = get_stock_summary(primary)
except Exception:
    summary = f"Summary for {primary} unavailable."

if primary != st.session_state.last_summary_ticker:
    add_to_history("bot", summary)
    st.session_state.last_summary_ticker = primary

try:
    tk_info = yf.Ticker(primary).info
    sector, industry = tk_info.get("sector", ""), tk_info.get("industry", "")
except Exception:
    tk_info, sector, industry = {}, "", ""

try:
    hist = yf.Ticker(primary).history(period="5d")["Close"]
    last_px = hist.iloc[-1]
    pct_px = (last_px - hist.iloc[-2]) / hist.iloc[-2] * 100
except Exception:
    last_px = pct_px = float("nan")

with st.sidebar:
    st.markdown("### 🧾 Snapshot", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='text-align:center;font-size:30px;font-weight:bold;color:white;'>${last_px:.2f}</div>
        <div style='text-align:center;font-size:16px;color:{"green" if pct_px>=0 else "red"};'>{pct_px:+.2f}%</div>
        <hr style='margin:10px 0;border:1px solid #333;'/>
        <div style='font-size:13px;'>Market Cap: <b>${tk_info.get('marketCap',0)/1e9:.2f} B</b></div>
        <div style='font-size:13px;'>P/E Ratio: <b>{tk_info.get('trailingPE','—')}</b></div>
        """,
        unsafe_allow_html=True,
    )

# ───────────────────────── Strategy Designer ─────────────────────────
st.subheader("📑 Strategy Designer")
sector_in = st.text_input("Sector focus", sector or industry or "")
goal      = st.selectbox("Positioning goal", ["Long", "Short", "Hedged", "Neutral"])
avoid_sym = st.text_input("Hedge / avoid ticker", primary)
capital   = st.number_input("Capital (USD)", 1000, step=1000, value=10000)
horizon   = st.slider("Time horizon (months)", 1, 24, 6)

with st.expander("⚖️ Risk Controls", False):
    beta_rng  = st.slider("Beta match band", 0.5, 1.5, (0.8, 1.2), 0.05)
    stop_loss = st.slider("Stop-loss for shorts (%)", 1, 20, 10)

if st.button("Suggest strategy"):
    basket_txt = ", ".join(basket)
    prompt = (
        f"Design a {goal.lower()} equity strategy using the basket [{basket_txt}]. "
        f"Sector focus: {sector_in}. Hedge or avoid exposure to {avoid_sym}. "
        f"Allocate a total of ${capital} over a {horizon}-month time horizon. "
        f"Match pair betas within {beta_rng[0]:.2f}-{beta_rng[1]:.2f}, "
        f"and apply a {stop_loss}% stop-loss to shorts.\n\n"
        "Return a markdown table with columns: Ticker, Position, Amount, Rationale, "
        "then a concise summary and 2-3 risk factors with explicit sources."
    )
    with st.spinner("Generating…"):
        plan = ask_openai(model, "You are a portfolio strategist.", prompt)

    st.markdown("### 📌 Suggested Strategy")
    st.write(plan)

    # Pull out the risk section (if any) for emphasis
    match = re.search(r"(### Risks.*?)(?=\n### |\Z)", plan, re.DOTALL | re.I)
    if match:
        st.markdown("### ⚠️ Highlighted Risks")
        st.markdown(
            f"<div class='card'><pre style='white-space:pre-wrap;font-size:14px;'>"
            f"{match.group(1).strip()}</pre></div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No specific risks cited by the model.")

# ───────────────────────── Price Charts (optional) ───────────────────
if show_charts:
    st.subheader("📈 Price Comparison")
    duration = st.selectbox("Duration", ["1mo", "3mo", "6mo", "1y"], 2, key="dur_sel")
    comps_selected = st.multiselect(
        "Tickers to plot", options=basket, default=basket, key="plot_sel"
    )
    if "SPY" not in comps_selected:
        comps_selected.append("SPY")

    price_df = fetch_stock_df(comps_selected, duration)
    if price_df.empty:
        st.error("No price data.")
    else:
        st.plotly_chart(
            px.line(price_df, title=f"Prices ({duration})",
                    labels={"value": "Price", "variable": "Ticker"}),
            use_container_width=True,
        )

        st.markdown("### 💹 Latest Prices")
        cols = st.columns(len(price_df.columns))
        for c, sym in zip(cols, price_df.columns):
            ser = price_df[sym]
            last = ser.iloc[-1]
            delta = ser.pct_change().iloc[-1] * 100
            with c:
                st.plotly_chart(
                    px.line(ser, height=80)
                    .update_layout(
                        showlegend=False,
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis=dict(showticklabels=False),
                        yaxis=dict(showticklabels=False),
                    ),
                    use_container_width=True,
                )
                st.markdown(
                    f"<div style='font-size:20px;font-weight:bold;'>{sym}</div>"
                    f"<div style='font-size:18px;'>${last:.2f}</div>"
                    f"<div style='color:{'green' if delta>=0 else 'red'};'>{delta:+.2f}%</div>",
                    unsafe_allow_html=True,
                )

# ─────────────────────────── Quick Chat ──────────────────────────────
st.markdown("---")
st.header("💬 Quick Chat (optional)")
for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

user_q = st.chat_input("Ask anything…")
if user_q:
    add_to_history("user", user_q)
    ctx = f"Summary: {summary}\nPortfolio: {', '.join(portfolio)}"
    ans = ask_openai(model, "You are a helpful market analyst.", ctx + "\n\n" + user_q)
    add_to_history("assistant", ans)
    st.rerun()
