# app.py  – streamlined stock-chatbot dashboard

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

from config import DEFAULT_MODEL
from stock_utils import get_stock_summary
from openai_client import ask_openai

#───────────────── 0. PAGE CONFIG ─────────────────#
st.set_page_config(page_title="Market Movement Chatbot", layout="wide")
st.title("📈 Market Movement Chatbot")

#───────────────── 1. SESSION STATE ───────────────#
if "history" not in st.session_state:
    st.session_state.history = []

def add_to_history(role: str, text: str):
    st.session_state.history.append((role, text))

#───────────────── 2. CACHED HELPERS ──────────────#
@st.cache_data(ttl=300)
def fetch_stock_df(tickers: list[str], period: str) -> pd.DataFrame:
    """Pull Close prices and drop tickers with no data."""
    df = yf.download(tickers, period=period, progress=False)["Close"]
    return df.dropna(axis=1, how="all")

@st.cache_data(ttl=300)
def fetch_competitors_llm(model: str, name: str, domain: str) -> list[str]:
    """Ask LLM for up to 7 competitor tickers and sanitize output."""
    prompt = (
        f"List ONLY the top 7 stock ticker symbols of public companies that compete "
        f"with {name} in the '{domain}' domain. Return a plain Python list, e.g. "
        f"['MSFT', 'GOOG', 'NVDA']."
    )
    resp = ask_openai(model, "You are a helpful stock analyst.", prompt)
    try:
        lst = eval(resp.strip(), {"__builtins__": None}, {})
        return [t.strip().upper() for t in lst if isinstance(t, str)]
    except Exception:
        lines = [ln.strip('",[] ') for ln in resp.splitlines()]
        return [ln.upper() for ln in lines if ln.isalpha() and 0 < len(ln) <= 5][:7]

#───────────────── 3. SIDEBAR: settings + snapshot ─#
with st.sidebar.expander("⚙️ Settings", expanded=False):
    model = st.selectbox(
        "OpenAI Model",
        options=[DEFAULT_MODEL, "gpt-4.1-mini", "gpt-4o-mini",
                 "gpt-3.5-turbo", "gpt-4", "gpt-4o"],
        index=0,
    )
    if st.button("🧹 Clear Chat History"):
        st.session_state.history = []

#───────────────── 4. MAIN TICKER INPUT ───────────#
ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA)", "AAPL").upper().strip()
if not ticker:
    st.stop()

#─ 4.a  LLM stock summary (stored in chat) ─#
summary = get_stock_summary(ticker)
add_to_history("bot", summary)

#─ 4.b  Basic metadata (handle YF rate-limits) ─#
try:
    info     = yf.Ticker(ticker).info
    sector   = info.get("sector", "")
    industry = info.get("industry", "")
    longname = info.get("longName", ticker)
except Exception:
    info = {}; sector = industry = ""; longname = ticker

#─ 4.c  Latest snapshot metrics ─#
try:
    prices5 = yf.Ticker(ticker).history(period="5d")["Close"]
    last_px = prices5.iloc[-1]
    pct_px  = (last_px - prices5.iloc[-2]) / prices5.iloc[-2] * 100
except Exception:
    last_px, pct_px = float("nan"), float("nan")

with st.sidebar:
    st.markdown("### ℹ️ Snapshot")
    st.metric("Price", f"${last_px:.2f}", f"{pct_px:+.2f}%")
    st.metric("Market Cap", f"${info.get('marketCap',0)/1e9:.1f} B")
    st.metric("P/E", str(info.get("trailingPE", "—")))

#───────────────── 5. DOMAIN & COMPETITORS ────────#
domains = [d for d in (sector, industry) if d] or ["General"]
domain_selected = st.selectbox("Which domain would you like to explore?", domains)
competitors_all = fetch_competitors_llm(model, longname, domain_selected)

#───────────────── 6. TABS ────────────────────────#
tab_compare, tab_strategy, tab_chat = st.tabs(["📈 Compare", "🎯 Strategy", "💬 Chat"])

#── 6.a  COMPARE TAB ──────────────────────────────#
with tab_compare:
    st.subheader("Price Comparison")

    comps_choice = st.multiselect(
        "Select competitors", options=competitors_all,
        default=competitors_all[:3]
    )
    duration = st.selectbox("Duration", ["1mo", "3mo", "6mo", "1y"], index=2)
    prices_df = fetch_stock_df([ticker] + comps_choice, duration)

    if prices_df.empty:
        st.error("No price data for selected symbols.")
    else:
        st.plotly_chart(
            px.line(prices_df, title=f"Prices ({duration})",
                    labels={"value":"Price","variable":"Ticker"}),
            use_container_width=True
        )

        st.markdown("### Latest Prices")
        cols = st.columns(len(prices_df.columns))
        for col, sym in zip(cols, prices_df):
            series = prices_df[sym]
            last   = series.iloc[-1]
            delta  = series.pct_change().iloc[-1] * 100
            spark  = px.line(series, height=80)
            spark.update_layout(
                showlegend=False, margin=dict(l=0,r=0,t=0,b=0),
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False)
            )
            col.plotly_chart(spark, use_container_width=True)
            col.metric(sym, f"${last:.2f}", f"{delta:+.2f}%")

#── 6.b  STRATEGY TAB ─────────────────────────────#
with tab_strategy:
    st.subheader("Strategy Assistant")

    # Prefill helpers
    default_sector  = sector or industry or ""
    default_avoid   = ticker

    # Core inputs
    sector_in   = st.text_input("Sector", default_sector, placeholder="e.g., AI")
    goal        = st.selectbox("Positioning Goal", ["Long", "Short", "Hedged", "Neutral"])
    avoid_sym   = st.text_input("Stock to hedge / avoid", default_avoid)

    # NEW: capital & horizon
    capital_usd = st.number_input("Capital to allocate (USD)", 1000, step=1000, value=10000)
    horizon_mo  = st.slider("Time horizon (months)", 1, 24, 6)

    # Risk controls
    with st.expander("⚖️ Risk Controls", expanded=False):
        beta_rng  = st.slider("Beta match (pair legs)", 0.5, 1.5, (0.8,1.2), 0.05)
        stop_loss = st.slider("Stop-loss for shorts (%)", 1, 20, 10)

    # Suggest strategy
    if st.button("Suggest Strategy"):
        user_intent = (
            f"Design a {goal.lower()} strategy in the {sector_in} sector. "
            f"Hedge/avoid {avoid_sym}. Allocate ${capital_usd} over {horizon_mo} months. "
            f"Pairs must have betas in {beta_rng[0]:.2f}-{beta_rng[1]:.2f}; "
            f"short legs get a {stop_loss}% stop-loss. "
            "Return 2-3 positions with dollar sizing and rationale."
        )
        with st.spinner("Generating strategy…"):
            plan = ask_openai(
                model,
                "You are a portfolio strategist. Output position table + narrative.",
                user_intent,
            )
        st.markdown("### 📌 Suggested Strategy")
        st.write(plan)

#── 6.c  CHAT TAB ─────────────────────────────────#
with tab_chat:
    for role, msg in st.session_state.history:
        st.chat_message(role).write(msg)

    q = st.chat_input("Ask a question…")
    if q:
        add_to_history("user", q)
        ctx = (
            f"Summary: {summary}\n"
            f"Domain: {domain_selected}\n"
            f"Competitors: {', '.join(competitors_all)}"
        )
        a = ask_openai(model,
                       "You are a helpful market analyst. Use context.",
                       ctx + "\n\n" + q)
        add_to_history("assistant", a)
        st.experimental_rerun()
