# app.py ─ Multi-ticker Market-Movement Chatbot
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

from config import DEFAULT_MODEL
from stock_utils import get_stock_summary
from openai_client import ask_openai

st.set_page_config(page_title="Market Movement Chatbot", layout="wide")
st.title("📈 Market Movement Chatbot")

# ─────────────────── Session state ───────────────────
if "history" not in st.session_state:
    st.session_state.history = []
def add_to_history(role, txt): st.session_state.history.append((role, txt))

# ─────────────────── Cached helpers ──────────────────
@st.cache_data(ttl=300)
def fetch_stock_df(tickers, period):
    df = yf.download(tickers, period=period, progress=False)["Close"]
    return df.dropna(axis=1, how="all")

@st.cache_data(ttl=300)
def fetch_competitors_llm(model, name, domain):
    prompt = (
        f"List ONLY the top 7 stock ticker symbols of companies that compete with {name} "
        f"in the '{domain}' domain. Return a Python list like ['MSFT','GOOG']."
    )
    resp = ask_openai(model, "You are a helpful stock analyst.", prompt)
    try:
        lst = eval(resp.strip(), {"__builtins__": None}, {})
        return [t.upper() for t in lst if isinstance(t, str)]
    except Exception:
        lines = [ln.strip('",[] ') for ln in resp.splitlines()]
        return [ln.upper() for ln in lines if ln.isalpha()][:7]

# ─────────────────── Sidebar: settings ───────────────
with st.sidebar.expander("⚙️ Settings", expanded=False):
    model = st.selectbox(
        "OpenAI Model",
        [DEFAULT_MODEL, "gpt-4.1-mini", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4o"],
        0,
    )
    if st.button("🧹 Clear Chat History"):
        st.session_state.history = []

# ─────────────────── Ticker input (basket) ───────────
tickers_raw = st.text_input(
    "Enter one or more tickers (comma-separated)", "AAPL, MSFT"
)
tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
if not tickers:
    st.stop()

primary = tickers[0]  # first symbol drives snapshot & sector

# ─────────────────── Snapshot & metadata ─────────────
summary = get_stock_summary(primary); add_to_history("bot", summary)
try:
    info = yf.Ticker(primary).info
    sector, industry = info.get("sector", ""), info.get("industry", "")
except Exception:
    info, sector, industry = {}, "", ""

try:
    hist = yf.Ticker(primary).history(period="5d")["Close"]
    last_px = hist.iloc[-1]; pct_px = (last_px - hist.iloc[-2]) / hist.iloc[-2] * 100
except Exception:
    last_px = pct_px = float("nan")

with st.sidebar:
    st.markdown("### ℹ️ Snapshot")
    st.metric("Price", f"${last_px:.2f}", f"{pct_px:+.2f}%")
    st.metric("Market Cap", f"${info.get('marketCap',0)/1e9:.1f} B")
    st.metric("P/E", str(info.get("trailingPE", "—")))

# ─────────────────── Domain + competitor logic ──────
domains = [d for d in (sector, industry) if d] or ["General"]
domain_selected = st.selectbox("Domain context", domains)

if len(tickers) == 1:
    competitors_all = fetch_competitors_llm(model, primary, domain_selected)
    basket = [primary] + competitors_all[:3]        # default compare list
else:
    competitors_all = tickers[1:]                   # treat extras as comps
    basket = tickers                                # compare exactly what user typed

# ─────────────────── Tabs ───────────────────────────
tab_compare, tab_strategy, tab_chat = st.tabs(["📈 Compare", "🎯 Strategy", "💬 Chat"])

# 1) Compare tab
with tab_compare:
    st.subheader("Price Comparison")
    comps_selected = st.multiselect("Select symbols to plot", basket, default=basket)
    duration = st.selectbox("Duration", ["1mo", "3mo", "6mo", "1y"], 2)
    price_df = fetch_stock_df(comps_selected, duration)
    if price_df.empty:
        st.error("No price data.")
    else:
        st.plotly_chart(
            px.line(price_df, title=f"Prices ({duration})",
                    labels={"value": "Price", "variable": "Ticker"}),
            use_container_width=True
        )
        st.markdown("### Latest Prices")
        cols = st.columns(len(price_df.columns))
        for c, sym in zip(cols, price_df.columns):
            ser, last, delta = price_df[sym], price_df[sym].iloc[-1], price_df[sym].pct_change().iloc[-1]*100
            spark = px.line(ser, height=80).update_layout(
                showlegend=False, margin=dict(l=0,r=0,t=0,b=0),
                xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False)
            )
            c.plotly_chart(spark, use_container_width=True)
            c.metric(sym, f"${last:.2f}", f"{delta:+.2f}%")

# 2) Strategy tab
with tab_strategy:
    st.subheader("Strategy Assistant")
    default_sector, default_avoid = sector or industry or "", primary
    sector_in = st.text_input("Sector focus", default_sector)
    goal      = st.selectbox("Positioning goal", ["Long", "Short", "Hedged", "Neutral"])
    avoid_sym = st.text_input("Hedge / avoid ticker", default_avoid)
    capital   = st.number_input("Capital (USD)", 1000, step=1000, value=10000)
    horizon   = st.slider("Time horizon (months)", 1, 24, 6)

    with st.expander("⚖️ Risk Controls", False):
        beta_rng  = st.slider("Beta match band", 0.5, 1.5, (0.8,1.2), 0.05)
        stop_loss = st.slider("Stop-loss for shorts (%)", 1, 20, 10)

    if st.button("Suggest Strategy"):
        basket_txt = ", ".join(comps_selected or [primary])
        prompt = (
            f"Design a {goal.lower()} strategy using the basket [{basket_txt}]. "
            f"Sector focus: {sector_in}. Hedge/avoid {avoid_sym}. "
            f"Allocate ${capital} over {horizon} months. "
            f"Pair betas within {beta_rng[0]:.2f}-{beta_rng[1]:.2f}; "
            f"shorts must carry a {stop_loss}% stop-loss. "
            "Return 2–3 positions with dollar sizing and rationale."
        )
        with st.spinner("Generating…"):
            plan = ask_openai(
                model,
                "You are a portfolio strategist. Output a table + narrative.",
                prompt,
            )
        st.markdown("### 📌 Suggested Strategy")
        st.write(plan)

# 3) Chat tab
with tab_chat:
    for role, msg in st.session_state.history:
        st.chat_message(role).write(msg)
    q = st.chat_input("Ask anything…")
    if q:
        add_to_history("user", q)
        ctx = f"Summary: {summary}\nDomain: {domain_selected}\nTickers: {', '.join(basket)}"
        ans = ask_openai(model, "You are a helpful market analyst.", ctx + "\n\n" + q)
        add_to_history("assistant", ans); st.experimental_rerun()
