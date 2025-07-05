# app.py – Market‑Movement Chatbot (single‑screen strategy edition)
"""Streamlit app that helps investors design equity strategies.

Main layout:
• Sidebar: settings, snapshot, quarterly outlook, optional price‑chart toggle
• Main page: stock selection, strategy designer, conditional charts, chat

All helper logic kept inside this single file for easy deployment to
Streamlit Cloud or local use.
"""

import math
import re
import textwrap
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

from config import DEFAULT_MODEL
from openai_client import ask_openai
from stock_utils import get_stock_summary

# ────────────────────────────── Set‑up ──────────────────────────────
st.set_page_config(page_title="Strategy Chatbot", layout="wide")

# ---------- Global CSS ----------
st.markdown(
    """
    <style>
    .card        {background:#1e1f24;padding:18px;border-radius:12px;margin-bottom:18px;}
    .metric-tile {background:#f1f5f90D;border:1px solid #33415550;padding:18px 22px;border-radius:12px;
                  transition:background .2s;cursor:pointer;}
    .metric-tile:hover{background:#33415522;}
    .metric-title{font-weight:600;font-size:18px;margin-bottom:6px;}
    .metric-value{font-size:22px;font-weight:700;}
    .chevron     {float:right;font-size:20px;line-height:18px;transform:translateY(2px);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎯 Equity Strategy Assistant")

# ───────────────────────── Session State ──────────────────────────
if "history" not in st.session_state:
    st.session_state.history: List = []
if "tickers_selected" not in st.session_state:
    st.session_state.tickers_selected: List[str] = ["AAPL", "MSFT"]
if "outlook_md" not in st.session_state:
    st.session_state.outlook_md: str | None = None

def add_to_history(role: str, txt: str) -> None:
    st.session_state.history.append((role, txt))

# ───────────────────────── Helper Functions ───────────────────────

def clean_llm_markdown(md: str) -> str:
    """Light clean‑up: remove stray emphasis and join words/digits."""
    md = re.sub(r"(\d)(?=[a-zA-Z])", r"\1 ", md)
    md = re.sub(r"([a-zA-Z])(?=\d)", r"\1 ", md)
    return md.replace("*", "").replace("_", "")


def quarters_sparkline(tk: yf.Ticker, metric: str) -> go.Figure:
    """Return a tiny Plotly figure for quarterly revenue or earnings."""
    df = None
    try:
        if hasattr(tk, "quarterly_earnings") and isinstance(tk.quarterly_earnings, pd.DataFrame):
            df = (
                tk.quarterly_earnings.copy()
                .reset_index()
                .rename(columns={"Quarter": "Quarter", "Earnings": "Value"})
            )
    except Exception:
        pass

    if df is None or df.empty:
        try:
            income_stmt = tk.income_stmt
            if isinstance(income_stmt, pd.DataFrame):
                if metric == "revenue" and "Total Revenue" in income_stmt.index:
                    df = income_stmt.loc["Total Revenue"].to_frame().reset_index()
                elif metric == "earnings" and "Net Income" in income_stmt.index:
                    df = income_stmt.loc["Net Income"].to_frame().reset_index()
                if df is not None:
                    df.columns = ["Quarter", "Value"]
        except Exception:
            df = pd.DataFrame()

    if df is None or df.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df["Quarter"], y=df["Value"], mode="lines+markers", line=dict(color="skyblue"))
    )
    fig.update_layout(height=160, margin=dict(t=10, l=10, r=10, b=10), xaxis_title=None, yaxis_title="Value")
    return fig


# ------------- small numeric extractor -------------

def grab_num(pattern: str, text: str) -> float:
    """Return the first captured float in *text* matching *pattern*.

    • Removes commas  • Handles optional $  • Converts “M” → billions
    • Returns np.nan if not found / bad float
    """
    m = re.search(pattern, text, re.I)
    if not m:
        return np.nan

    raw = m.group(1).replace(",", "")
    try:
        val = float(raw)
    except ValueError:
        return np.nan

    # Look right after the number for a unit suffix
    suffix = re.search(rf"{re.escape(raw)}\s*([MB])", text[m.end() - 1 : m.end() + 2], re.I)
    if suffix and suffix.group(1).upper() == "M":
        val /= 1_000  # convert millions → billions

    return val


# ------------- Yahoo ticker search -------------

@st.cache_data(ttl=3600, show_spinner=False)
def search_ticker_symbols(query: str, limit: int = 10):
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "quotesCount": limit, "newsCount": 0, "lang": "en"}
    try:
        resp = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=4)
        resp.raise_for_status()
        quotes = resp.json().get("quotes", [])
    except Exception:
        return []

    return [
        {"symbol": q.get("symbol", "").upper(), "name": q.get("shortname") or q.get("longname") or ""}
        for q in quotes
        if q.get("symbol")
    ]


# ------------- yfinance helpers -------------

@st.cache_data(ttl=300)
def fetch_stock_df(tickers: List[str], period: str) -> pd.DataFrame:
    df = yf.download(tickers, period=period, progress=False)["Close"]
    return df.dropna(axis=1, how="all")


@st.cache_data(ttl=300)
def fetch_competitors_llm(model: str, name: str, domain: str) -> List[str]:
    prompt = (
        f"List ONLY the top 7 stock ticker symbols of companies that compete with {name} "
        f"in the '{domain}' domain. Return a Python list like ['MSFT','GOOG']."
    )
    resp = ask_openai(model, "You are a helpful stock analyst.", prompt)
    try:
        import ast

        lst = ast.literal_eval(resp.strip())
        return [t.upper() for t in lst if isinstance(t, str)]
    except Exception:
        lines = [ln.strip('",[] ') for ln in resp.splitlines()]
        return [ln.upper() for ln in lines if ln.isalpha()][:7]


# ───────────────────────── Sidebar – Settings ────────────────────
with st.sidebar.expander("⚙️ Settings", expanded=False):
    model = st.selectbox("OpenAI Model", [DEFAULT_MODEL, "gpt-4.1-mini", "gpt-4o-mini"], 0)
    if st.button("🧹 Clear Chat History"):
        st.session_state.history = []
    if st.button("🛑 Clear Tickers"):
        st.session_state.tickers_selected = []

# Sidebar toggles
show_charts = st.sidebar.checkbox("📈 Show Price Charts", value=False)

# ───────────── Quarterly Outlook expander ─────────────
with st.sidebar.expander("🔮 Quarterly Outlook", expanded=False):
    if st.button("↻ Generate Outlook", key="btn_outlook"):
        st.session_state.outlook_md = None  # force refresh

    if st.session_state.outlook_md is None:
        # first pass – schedule generation and rerun
        st.session_state.outlook_md = "Generating…"
        st.experimental_rerun()

    elif st.session_state.outlook_md == "Generating…":
        # second pass – actually call the LLM
        primary_tmp = st.session_state.tickers_selected[0]
        prompt = (
            f"Provide numeric forecasts for **EPS** and **Total Revenue** for {primary_tmp}'s next quarter. "
            f"Include your prediction, Street consensus, and beat probability in %. "
            f"Add one sentence of reasoning ending with 'Source: …'. "
            f"Return in markdown: a table plus bullets, no code fences."
        )
        with st.spinner("Contacting LLM…"):
            raw_md = ask_openai(model, "You are a senior equity analyst, precise and data-driven.", prompt)
        st.session_state.outlook_md = clean_llm_markdown(raw_md)
        st.experimental_rerun()

    else:
        st.markdown(f"<div class='card'>{st.session_state.outlook_md}</div>", unsafe_allow_html=True)

# ───────────────────────── Stock Selection UI ───────────────────
st.markdown("### 📌 Stock Selection")
col1, col2 = st.columns([3, 2])

with col1:
    search_q = st.text_input("Search company or ticker", "", key="search_box")

with col2:
    tickers = st.session_state.tickers_selected
    primary = tickers[0] if tickers else None
    if tickers:
        primary = st.selectbox("Primary ticker", options=tickers, index=0, key="primary_select")

# Autocomplete suggestions
if len(search_q) >= 2:
    matches = search_ticker_symbols(search_q)
    if matches:
        display_opts = [f"{m['name']} ({m['symbol']})" for m in matches]
        choice = st.selectbox("Suggestions", display_opts, index=0, key="suggest_box")
        if st.button("➕ Add", key="add_btn"):
            sym = choice.split("(")[-1].rstrip(")")
            default_seed = {"AAPL", "MSFT"}
            if set(tickers) == default_seed:
                tickers.clear()
            if sym not in tickers:
                tickers.insert(0, sym)
            st.rerun()
    else:
        st.info("No matches yet… keep typing")

if not tickers:
    st.info("Add at least one ticker to proceed.")
    st.stop()

# ───────────────────────── Snapshot & Metadata ───────────────────
summary = get_stock_summary(primary)
add_to_history("bot", summary)

try:
    info = yf.Ticker(primary).info
    sector, industry = info.get("sector", ""), info.get("industry", "")
except Exception:
    info, sector, industry = {}, "", ""

try:
    hist = yf.Ticker(primary).history(period="5d")["Close"]
    last_px = hist.iloc[-1]
    pct_px = (last_px - hist.iloc[-2]) / hist.iloc[-
