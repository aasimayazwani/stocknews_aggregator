# app.py – Market-Movement Chatbot  (portfolio-aware + risk-scan edition)
from __future__ import annotations

import re, textwrap, requests
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from config import DEFAULT_MODEL          # local module
from openai_client import ask_openai      # wrapper around OpenAI API
from stock_utils import get_stock_summary # your own helper
# ────────────────────────────────── THEME ─────────────────────────────────
st.set_page_config(page_title="Strategy Chatbot", layout="wide")
st.markdown(
    """
    <style>
      .card{background:#1e1f24;padding:18px;border-radius:12px;margin-bottom:18px;}
      .chip{display:inline-block;margin:0 6px 6px 0;padding:4px 10px;border-radius:14px;
            background:#33415588;color:#f1f5f9;font-weight:600;font-size:13px;}
      .metric{font-size:18px;font-weight:600;margin-bottom:2px;}
      .metric-small{font-size:14px;}
      label{font-weight:600;font-size:0.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🎯  Equity Strategy Assistant")

# ─────────────────────────────── STATE ────────────────────────────────
if "history"     not in st.session_state: st.session_state.history     = []
if "portfolio"   not in st.session_state: st.session_state.portfolio   = ["AAPL", "MSFT"]
if "outlook_md"  not in st.session_state: st.session_state.outlook_md  = None
if "risk_cache"  not in st.session_state: st.session_state.risk_cache  = {}  # {ticker: [risks]}
if "risk_ignore" not in st.session_state: st.session_state.risk_ignore = []  # selected exclusions

# ──────────────────────────── HELPERS ────────────────────────────────
def clean_md(md: str) -> str:
    md = re.sub(r"(\d)(?=[A-Za-z])", r"\1 ", md)
    return md.replace("*", "").replace("_", "")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers: List[str], period="2d"):
    df = yf.download(tickers, period=period, progress=False)["Close"]
    return df.dropna(axis=1, how="all")

@st.cache_data(ttl=900, show_spinner=False)
def web_risk_scan(ticker: str) -> List[str]:
    """Very light-weight DuckDuckGo instant-answer style scrape for ‘<ticker> stock risks’."""
    try:
        r = requests.get(
            "https://api.duckduckgo.com",
            params={"q": f"{ticker} stock risks", "format": "json", "no_redirect": "1"},
            timeout=6,
        )
        r.raise_for_status()
        rel = r.json().get("RelatedTopics", [])[:8]
        texts = [t.get("Text", "") for t in rel if isinstance(t, dict) and t.get("Text")]
        # basic clean-up
        dedup = list(dict.fromkeys([re.sub(r"\.$", "", t.strip()) for t in texts if t.strip()]))
        # fallback
        return dedup or [f"No obvious current headline risks detected for {ticker}."]
    except Exception:
        return [f"Could not retrieve risk headlines for {ticker}."]

# ────────────────────────── SIDEBAR – SETTINGS ───────────────────────
with st.sidebar.expander("⚙️  Settings"):
    model = st.selectbox("OpenAI Model", [DEFAULT_MODEL, "gpt-4.1-mini", "gpt-4o-mini"], 0)
    if st.button("🧹  Clear chat history"):  st.session_state.history = []
    if st.button("🗑️  Clear portfolio"):    st.session_state.portfolio = []

show_charts = st.sidebar.checkbox("📈  Show compar-chart", value=False)

# ───────────────────────────── PORTFOLIO UI ──────────────────────────
raw = st.text_input("Tickers you hold (comma-separated)", ", ".join(st.session_state.portfolio))
portfolio = [t.strip().upper() for t in raw.split(",") if t.strip()]
st.session_state.portfolio = portfolio
if not portfolio:
    st.error("Enter at least one ticker to continue."); st.stop()

# Chips + price tiles
st.markdown("".join(f"<span class='chip'>{t}</span>" for t in portfolio), unsafe_allow_html=True)
tiles_df = fetch_prices(portfolio, "2d")
if not tiles_df.empty:
    last, prev = tiles_df.iloc[-1], tiles_df.iloc[-2]
    cols = st.columns(len(tiles_df.columns))
    for c, sym in zip(cols, tiles_df.columns):
        v, d = last[sym], (last[sym]-prev[sym]) / prev[sym] * 100
        with c:
            c.markdown(f"<div class='metric'>{sym}</div>", unsafe_allow_html=True)
            c.markdown(f"{v:,.2f}")
            c.markdown(
                f"<span class='metric-small' style='color:{'lime' if d>=0 else 'tomato'}'>{d:+.2f}%</span>",
                unsafe_allow_html=True,
            )

st.divider()
primary = st.selectbox("🎯  Focus stock", portfolio, 0)
others  = [t for t in portfolio if t != primary]
basket  = [primary] + others

# ────────────────────── AUTOMATED RISK SCAN SECTION ───────────────────
st.markdown("### 🔍  Key headline risks")

if primary not in st.session_state.risk_cache:
    with st.spinner("Scanning web…"):
        st.session_state.risk_cache[primary] = web_risk_scan(primary)

risk_list = st.session_state.risk_cache[primary]
exclude   = st.multiselect(
    "Un-check any headline you **do not** want the LLM to consider:",
    options=risk_list,
    default=risk_list,
    key="risk_select",
)
st.session_state.risk_ignore = [r for r in risk_list if r not in exclude]

# ───────────────────────── SIDEBAR – OUTLOOK ─────────────────────────
with st.sidebar.expander("🔮  Quarterly outlook"):
    if st.button("↻  Refresh forecast"): st.session_state.outlook_md = None
    if st.session_state.outlook_md is None:
        st.session_state.outlook_md = "Generating…"; st.rerun()
    elif st.session_state.outlook_md == "Generating…":
        p = (
            f"Provide EPS and total-revenue forecasts for {primary}'s next quarter; "
            f"include Street consensus and beat probability (in %). End with 'Source: …'. "
            f"Return markdown (table + bullets)."
        )
        md = ask_openai(model, "You are a senior equity analyst.", p)
        st.session_state.outlook_md = clean_md(md); st.rerun()
    else:
        st.markdown(f"<div class='card'>{st.session_state.outlook_md}</div>", unsafe_allow_html=True)

# ─────────────────────────── STRATEGY DESIGNER ───────────────────────
st.markdown("### 📝  Strategy Designer")
sector_guess = yf.Ticker(primary).info.get("sector", "")
sector_in    = st.text_input("Sector", sector_guess)
goal         = st.selectbox("Positioning goal", ["Long", "Short", "Hedged", "Neutral"])
avoid_sym    = st.text_input("Hedge / avoid ticker", primary)
capital      = st.number_input("Capital (USD)", 1000, 1_000_000, 10_000, 1000)
horizon      = st.slider("Time horizon (months)", 1, 24, 6)

with st.expander("⚖️  Risk controls"):
    beta_rng  = st.slider("Beta match band", 0.5, 1.5, (0.8, 1.2), 0.05)
    stop_loss = st.slider("Stop-loss for shorts (%)", 1, 20, 10)

if st.button("Suggest strategy", type="primary"):
    ignore_txt = "; ".join(st.session_state.risk_ignore)
    prompt = textwrap.dedent(f"""
        Design a {goal.lower()} strategy around the basket [{', '.join(basket)}].
        Sector focus: {sector_in}. Hedge/avoid: {avoid_sym}.
        Capital: ${capital:,.0f}; Horizon: {horizon} months.
        Keep pair betas in {beta_rng[0]:.2f}-{beta_rng[1]:.2f}; stop-loss {stop_loss}%.
        The following headline risks have been detected for {primary}: {', '.join(risk_list) or 'None'}.
        EXCLUDE the following risks from consideration: {ignore_txt or 'None'}.
        Return markdown with a table (Ticker | Position | Amount | Rationale),
        a short summary, and 2-3 explicit residual risks with citations.
    """).strip()

    with st.spinner("Calling LLM…"):
        plan = ask_openai(model, "You are a portfolio strategist.", prompt)

    st.subheader("📌  Suggested strategy")
    st.write(plan)

    if m := re.search(r"(### Risks.*?)(?=\n### |\Z)", plan, re.I | re.S):
        st.subheader("⚠️  Highlighted Risks")
        st.markdown(f"<div class='card'>{m.group(1).strip()}</div>", unsafe_allow_html=True)

# ─────────────────────────── OPTIONAL CHARTS ─────────────────────────
if show_charts:
    st.markdown("### 📈  Price comparison")
    duration = st.selectbox("Duration", ["1mo", "3mo", "6mo", "1y"], 2)
    plot_tickers = st.multiselect("Tickers to plot", basket + ["SPY"], basket)
    if "SPY" not in plot_tickers: plot_tickers.append("SPY")
    chart_df = fetch_prices(plot_tickers, duration)
    if chart_df.empty:
        st.error("No price data.")
    else:
        st.plotly_chart(
            px.line(chart_df, title=f"Adjusted close ({duration})",
                    labels={"value": "Price", "variable": "Ticker"}),
            use_container_width=True,
        )

# ───────────────────────────── QUICK CHAT ────────────────────────────
st.divider()
st.markdown("### 💬  Quick chat")
for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

if q := st.chat_input("Ask anything…"):
    ctx = f"User portfolio: {', '.join(portfolio)}. Focus: {primary}."
    st.session_state.history.append(("user", q))
    ans = ask_openai(model, "You are a helpful market analyst.", ctx + "\n\n" + q)
    st.session_state.history.append(("assistant", ans))
    st.rerun()
