# app.py – Market-Movement Chatbot (portfolio-aware UX refresh)
from __future__ import annotations

import re, textwrap
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
# ──────────────────────────────── THEME ───────────────────────────────
st.set_page_config(page_title="Strategy Chatbot", layout="wide")

st.markdown(
    """
    <style>
      .card{background:#1e1f24;padding:18px;border-radius:12px;margin-bottom:18px;}
      .chip{display:inline-block;margin:0 6px 6px 0;padding:4px 10px;
            border-radius:14px;font-size:13px;font-weight:600;
            background:#33415588;color:#f1f5f9;}
      .metric{font-size:18px;font-weight:600;margin-bottom:2px;}
      .metric-small{font-size:14px;}
      /* tidy default widgets */
      label{font-weight:600;font-size:0.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🎯 Equity Strategy Assistant")

# ───────────────────────────── Session State ──────────────────────────
if "history" not in st.session_state:
    st.session_state.history: List = []
if "portfolio" not in st.session_state:
    st.session_state.portfolio: List[str] = ["AAPL", "MSFT"]
if "outlook_md" not in st.session_state:
    st.session_state.outlook_md: str | None = None

def clean_llm_markdown(md: str) -> str:
    md = re.sub(r"(\d)(?=[A-Za-z])", r"\1 ", md)
    md = re.sub(r"([A-Za-z])(?=\d)", r"\1 ", md)
    return md.replace("*", "").replace("_", "")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_df(tks: List[str], period: str) -> pd.DataFrame:
    df = yf.download(tks, period=period, progress=False)["Close"]
    return df.dropna(axis=1, how="all")

# ───────────────────────── Sidebar – Settings ─────────────────────────
with st.sidebar.expander("⚙️  Settings"):
    model = st.selectbox("OpenAI model", [DEFAULT_MODEL, "gpt-4.1-mini", "gpt-4o-mini"], 0)
    if st.button("🧹 Clear chat history"):
        st.session_state.history = []
    if st.button("🗑️  Clear portfolio"):
        st.session_state.portfolio = []

show_charts = st.sidebar.checkbox("Display comparison chart", False)

# ───────────────────────────── PORTFOLIO UI ──────────────────────────
st.markdown("### 💼 Your Portfolio")

raw_input = st.text_input(
    "Tickers (comma-separated)",
    value=", ".join(st.session_state.portfolio),
)
portfolio = [t.strip().upper() for t in raw_input.split(",") if t.strip()]
st.session_state.portfolio = portfolio

if not portfolio:
    st.error("Enter at least one ticker to continue.")
    st.stop()

# Inline chips for quick visual feedback
st.markdown(
    "".join(f"<span class='chip'>{t}</span>" for t in portfolio),
    unsafe_allow_html=True,
)

# Quick table of latest prices
price_df = fetch_stock_df(portfolio, "2d")
if not price_df.empty:
    last_row, prev_row = price_df.iloc[-1], price_df.iloc[-2]
    cols = st.columns(len(price_df.columns))
    for c, sym in zip(cols, price_df.columns):
        last, delta = last_row[sym], (last_row[sym] - prev_row[sym]) / prev_row[sym] * 100
        with c:
            st.markdown(f"<div class='metric'>{sym}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric'>{last:,.2f}</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='metric-small' style='color:{'lime' if delta>=0 else 'tomato'};'>{delta:+.2f}%</div>",
                unsafe_allow_html=True,
            )
st.divider()

# ─────────────────────── PRIMARY-FOCUS SELECTION ─────────────────────
primary = st.selectbox("🎯 Choose the stock to focus on", portfolio, 0)
other_holdings = [t for t in portfolio if t != primary]
basket = [primary] + other_holdings          # sent to LLM

# ─────────────────────── Sidebar – Quarterly Outlook ──────────────────
with st.sidebar.expander("🔮 Quarterly outlook"):
    if st.button("Generate outlook"):
        st.session_state.outlook_md = None

    if st.session_state.outlook_md is None:
        st.session_state.outlook_md = "Generating…"
        st.rerun()

    elif st.session_state.outlook_md == "Generating…":
        prompt = (
            f"Give EPS and revenue forecasts for {primary}'s next quarter; "
            f"include Street consensus and beat probability (in %). End with 'Source: …'. "
            f"Return markdown (table + bullets)."
        )
        with st.spinner("Contacting LLM…"):
            raw = ask_openai(model, "You are a senior equity analyst.", prompt)
        st.session_state.outlook_md = clean_llm_markdown(raw)
        st.rerun()

    else:
        st.markdown(f"<div class='card'>{st.session_state.outlook_md}</div>", unsafe_allow_html=True)

# ────────────────────────── STRATEGY DESIGNER ─────────────────────────
st.markdown("### 📝 Strategy Designer")
sector_guess = yf.Ticker(primary).info.get("sector", "")
sector_in = st.text_input("Sector", value=sector_guess)
goal      = st.selectbox("Position", ["Long", "Short", "Hedged", "Neutral"])
avoid_sym = st.text_input("Hedge / avoid ticker", value=primary)
capital   = st.number_input("Capital (USD)", 1000, 1_000_000, 10_000, 1000)
horizon   = st.slider("Time horizon (months)", 1, 24, 6)

with st.expander("Risk controls"):
    beta_rng  = st.slider("Beta match", 0.5, 1.5, (0.8, 1.2), 0.05)
    stop_loss = st.slider("Stop-loss for shorts (%)", 1, 20, 10)

if st.button("Suggest strategy", type="primary"):
    prompt = textwrap.dedent(
        f"""
        Design a {goal.lower()} strategy around [{', '.join(basket)}] given the user’s portfolio.
        *Sector focus*: {sector_in}. *Hedge/avoid*: {avoid_sym}.
        Allocate **${capital:,.0f}** over **{horizon} months**.
        Keep pair betas between {beta_rng[0]:.2f}-{beta_rng[1]:.2f}; stop-loss for shorts {stop_loss}%.
        Return a markdown table: Ticker | Position | Amount | Rationale,
        then a short summary plus 2-3 explicit risk factors with sources.
        """
    ).strip()

    with st.spinner("Generating…"):
        plan = ask_openai(model, "You are a portfolio strategist.", prompt)

    st.subheader("📌 Suggested strategy")
    st.write(plan)

    # Pull out risk block (if any)
    m = re.search(r"(### Risks.*?)(?=\n### |\Z)", plan, flags=re.I | re.S)
    if m:
        st.subheader("⚠️  Key risks")
        st.markdown(f"<div class='card'>{m.group(1).strip()}</div>", unsafe_allow_html=True)

# ────────────────────────── OPTIONAL CHARTS ──────────────────────────
if show_charts:
    st.markdown("### 📈 Price comparison")
    period = st.selectbox("Duration", ["1mo", "3mo", "6mo", "1y"], 2)
    plot_tickers = st.multiselect("Tickers to plot", basket + ["SPY"], basket)

    if "SPY" not in plot_tickers:
        plot_tickers.append("SPY")

    df = fetch_stock_df(plot_tickers, period)
    if df.empty:
        st.error("No price data.")
    else:
        st.plotly_chart(
            px.line(df, title=f"Adjusted close ({period})",
                    labels={"value": "Price", "variable": "Ticker"}),
            use_container_width=True,
        )

# ─────────────────────────── QUICK CHAT ──────────────────────────────
st.divider()
st.markdown("### 💬 Quick chat")
for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

if q := st.chat_input("Ask anything…"):
    # context to keep the model grounded in portfolio
    ctx = f"Portfolio: {', '.join(portfolio)}. Focus stock: {primary}."
    st.session_state.history.append(("user", q))
    ans = ask_openai(model, "You are a helpful market analyst.", ctx + "\n\n" + q)
    st.session_state.history.append(("assistant", ans))
    st.rerun()
