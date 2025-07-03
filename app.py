import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import time

from config import DEFAULT_MODEL
from stock_utils import get_stock_summary
from openai_client import ask_openai

st.set_page_config(page_title="Market Movement Chatbot", layout="wide")
st.title("📈 Market Movement Chatbot")

# ────────────────────────────────────────────────────────────────
# Session-state helpers
if "history" not in st.session_state:
    st.session_state.history = []

def add_to_history(role: str, text: str):
    st.session_state.history.append((role, text))

# ────────────────────────────────────────────────────────────────
# Caching helpers
@st.cache_data(ttl=300)
def fetch_stock_df(tickers: list[str], period: str) -> pd.DataFrame:
    """Download historical Close prices and drop empty columns."""
    df = yf.download(tickers, period=period, progress=False)["Close"]
    return df.dropna(axis=1, how="all")

@st.cache_data(ttl=300)
def fetch_competitors_llm(model: str, name: str, domain: str) -> list[str]:
    """Ask the LLM for up to 7 competitor tickers (plain Python list)."""
    prompt = (
        f"List ONLY the top 7 stock ticker symbols of companies that compete with {name} "
        f"in the '{domain}' domain. Return a plain Python list like ['MSFT', 'GOOG']."
    )
    resp = ask_openai(model, "You are a helpful stock analyst.", prompt)
    try:
        tickers = eval(resp.strip(), {"__builtins__": None}, {})
        return [t.strip().upper() for t in tickers if isinstance(t, str)]
    except Exception:
        return [
            line.strip().split()[0].strip('",[]').upper()
            for line in resp.splitlines() if line.strip()
        ]

# ────────────────────────────────────────────────────────────────
# Sidebar: Settings + Snapshot
with st.sidebar.expander("⚙️ Settings", expanded=False):
    model = st.selectbox(
        "OpenAI Model",
        options=[
            DEFAULT_MODEL, "gpt-4.1-mini", "gpt-4o-mini",
            "gpt-3.5-turbo", "gpt-4", "gpt-4o"
        ],
        index=0,
    )
    if st.button("🧹 Clear Chat"):
        st.session_state.history = []

# ────────────────────────────────────────────────────────────────
# Main ticker input
ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA)", "AAPL").upper().strip()
if not ticker:
    st.stop()

# ── Basic LLM summary (we keep this in chat history) ──
summary = get_stock_summary(ticker)
add_to_history("bot", summary)

# ── Fetch metadata & snapshot values (may hit YF rate limit → try/except) ──
try:
    info = yf.Ticker(ticker).info
    company_name = info.get("longName", ticker)
    sector = info.get("sector", "")
    industry = info.get("industry", "")
except Exception:
    info, company_name, sector, industry = {}, ticker, "", ""

# Latest price change for Snapshot
try:
    hist5 = yf.Ticker(ticker).history(period="5d")
    latest_price = hist5["Close"].iloc[-1]
    pct_change   = (latest_price - hist5["Close"].iloc[-2]) / hist5["Close"].iloc[-2] * 100
except Exception:
    latest_price, pct_change = float("nan"), float("nan")

# ── Sidebar Snapshot ──
with st.sidebar:
    st.markdown("### ℹ️ Snapshot")
    st.metric("Price", f"${latest_price:.2f}", f"{pct_change:+.2f}%")
    st.metric("Market Cap", f"${info.get('marketCap',0)/1e9:.1f} B")
    st.metric("P/E", str(info.get("trailingPE", "—")))

# ────────────────────────────────────────────────────────────────
# Derive “domain” for LLM competitor list
domains = [d for d in (sector, industry) if d] or ["General"]
domain_selected = st.selectbox("Which domain would you like to explore?", domains)

# Competitor tickers from LLM
competitors_all = fetch_competitors_llm(model, company_name, domain_selected)

# ────────────────────────────────────────────────────────────────
# Main tabs: Compare, Strategy, Chat
tab_compare, tab_strategy, tab_chat = st.tabs(
    ["📈 Compare", "🎯 Strategy", "💬 Chat"]
)

# ───────────── Compare Tab ─────────────
with tab_compare:
    st.subheader("📈 Compare Price Movement")

    # Controls
    comps_selected = st.multiselect(
        "Select competitors",
        options=competitors_all,
        default=competitors_all[:3]
    )
    duration = st.selectbox("Duration", ["1mo", "3mo", "6mo", "1y"], index=2)

    symbols = [ticker] + comps_selected
    df = fetch_stock_df(symbols, duration)

    # Validate
    if df.empty:
        st.error("No valid price data found for selected symbols.")
    else:
        st.plotly_chart(
            px.line(df, title=f"Price Comparison ({duration})", labels={"value": "Price", "variable": "Ticker"}),
            use_container_width=True
        )

        st.markdown("### 💹 Latest Price & % Change")
        valid_syms = df.columns.tolist()
        cols = st.columns(len(valid_syms))
        for c, sym in zip(cols, valid_syms):
            series = df[sym]
            last   = series.iloc[-1]
            delta  = series.pct_change().iloc[-1] * 100
            spark  = px.line(series, height=80)
            spark.update_layout(
                showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False)
            )
            c.plotly_chart(spark, use_container_width=True)
            c.metric(sym, f"${last:.2f}", f"{delta:+.2f}%")

# ───────────── Strategy Tab ─────────────
with tab_strategy:
    st.subheader("🎯 Strategy Assistant")

    # Prefills
    default_sector  = sector or industry or ""
    default_concern = ticker

    # Inputs
    sector_input = st.text_input(
        "Sector you're interested in",
        value=default_sector,
        placeholder="e.g., EV, AI, Semiconductors"
    )
    goal = st.selectbox("Positioning goal", ["Long", "Short", "Hedged", "Neutral"])
    concern = st.text_input(
        "Any stock to hedge/avoid?",
        value=default_concern,
        placeholder="e.g., TSLA"
    )

    # Advanced risk controls
    with st.expander("⚖️ Advanced Risk Controls", expanded=False):
        beta_range = st.slider(
            "Target beta range (pair legs)",
            min_value=0.5, max_value=1.5, value=(0.8, 1.2), step=0.05
        )
        stop_loss = st.slider(
            "Stop-loss for short legs (%)",
            min_value=1, max_value=20, value=10, step=1
        )

    # Suggest button
    if st.button("Suggest Strategy"):
        user_intent = (
            f"I want a {goal.lower()} strategy in the {sector_input} sector. "
            f"Hedge/avoid: {concern}. "
            f"Long/short legs must have betas within {beta_range[0]:.2f}-{beta_range[1]:.2f}. "
            f"Each short leg must include a {stop_loss}% stop-loss. "
            "Suggest 2-3 stock or ETF positions with rationale."
        )
        with st.spinner("Thinking…"):
            strat = ask_openai(
                model,
                "You are a portfolio strategist. Provide thoughtful long/short ideas "
                "that respect the beta and stop-loss constraints.",
                user_intent,
            )
        st.markdown("### 📌 Suggested Strategy")
        st.write(strat)

# ───────────── Chat Tab ─────────────
with tab_chat:
    for role, text in st.session_state.history:
        st.chat_message(role).write(text)

    q = st.chat_input("Ask a question…")
    if q:
        add_to_history("user", q)
        context = (
            f"Summary: {summary}\n"
            f"Domain: {domain_selected}\n"
            f"Competitors: {', '.join(competitors_all)}\n"
        )
        with st.spinner("Thinking…"):
            answer = ask_openai(
                model,
                "You are a helpful financial analyst assistant. Use the context.",
                context + "\n\n" + q
            )
        add_to_history("assistant", answer)
        st.experimental_rerun()
