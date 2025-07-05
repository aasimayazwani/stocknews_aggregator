# app.py ─ Market-Movement Chatbot (single-screen strategy edition)
import streamlit as st
import requests, yfinance as yf
import pandas as pd
import textwrap, re, math
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import Dict

from config import DEFAULT_MODEL
from stock_utils import get_stock_summary
from openai_client import ask_openai

st.set_page_config(page_title="Strategy Chatbot", layout="wide")

# ────────────────────────── Global CSS ───────────────────────────
st.markdown("""
<style>
.card{background:#1e1f24;padding:18px;border-radius:12px;margin-bottom:18px;}
.metric-tile{background:#f1f5f90D;border:1px solid #33415550;padding:18px 22px;border-radius:12px;
             transition:background .2s;cursor:pointer;}
.metric-tile:hover{background:#33415522;}
.metric-title{font-weight:600;font-size:18px;margin-bottom:6px;}
.metric-value{font-size:22px;font-weight:700;}
.chevron{float:right;font-size:20px;line-height:18px;transform:translateY(2px);}
</style>
""", unsafe_allow_html=True)

st.title("🎯 Equity Strategy Assistant")

# ───────────────────────── Session state ─────────────────────────
if "history" not in st.session_state:            # chat history
    st.session_state.history = []
if "tickers_selected" not in st.session_state:   # starter basket
    st.session_state.tickers_selected = ["AAPL", "MSFT"]
if "outlook_md" not in st.session_state:         # cached outlook text
    st.session_state.outlook_md = None

def add_to_history(role, txt):
    st.session_state.history.append((role, txt))

# ───────────────────────── Helper functions ──────────────────────
def clean_llm_markdown(md: str) -> str:
    md = re.sub(r"(\d)(?=[a-zA-Z])", r"\1 ", md)
    md = re.sub(r"([a-zA-Z])(?=\d)", r"\1 ", md)
    return md.replace("*", "").replace("_", "")

def quarters_sparkline(tk, metric: str):
    df = None
    try:
        if hasattr(tk, "quarterly_earnings") and isinstance(tk.quarterly_earnings, pd.DataFrame):
            df = tk.quarterly_earnings.copy().reset_index().rename(
                columns={"Quarter": "Quarter", "Earnings": "Value"})
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
    fig.add_trace(go.Scatter(x=df["Quarter"], y=df["Value"],
                             mode="lines+markers", line=dict(color="skyblue")))
    fig.update_layout(height=160, margin=dict(t=10, l=10, r=10, b=10),
                      xaxis_title=None, yaxis_title="Value")
    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def search_ticker_symbols(query: str, limit: int = 10):
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "quotesCount": limit, "newsCount": 0, "lang": "en"}
    try:
        resp = requests.get(url, params=params,
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=4)
        resp.raise_for_status()
        quotes = resp.json().get("quotes", [])
    except Exception:
        return []
    return [{"symbol": q.get("symbol", "").upper(),
             "name": q.get("shortname") or q.get("longname") or ""}
            for q in quotes if q.get("symbol")]

@st.cache_data(ttl=300)
def fetch_stock_df(tickers, period):
    df = yf.download(tickers, period=period, progress=False)["Close"]
    return df.dropna(axis=1, how="all")

@st.cache_data(ttl=300)
def fetch_competitors_llm(model, name, domain):
    prompt = (f"List ONLY the top 7 stock ticker symbols of companies that compete with {name} "
              f"in the '{domain}' domain. Return a Python list like ['MSFT','GOOG'].")
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
    model = st.selectbox("OpenAI Model",
                         [DEFAULT_MODEL, "gpt-4.1-mini", "gpt-4o-mini"], 0)
    if st.button("🧹 Clear Chat History"):
        st.session_state.history = []
    if st.button("🛑 Clear Tickers"):
        st.session_state.tickers_selected = []

# ─────────────── Sidebar – Show Charts + Quarterly Outlook ───────
show_charts = st.sidebar.checkbox("📈 Show Price Charts", value=False)

with st.sidebar.expander("🔮 Quarterly Outlook", expanded=False):
    if st.button("↻ Generate Outlook", key="btn_outlook"):
        st.session_state.outlook_md = None   # force refresh next run
    # Generate only if not cached
    if st.session_state.outlook_md is None:
        st.session_state.outlook_md = "Generating…"   # placeholder
        st.experimental_rerun()
    elif st.session_state.outlook_md == "Generating…":
        # First pass after click: actually call the model
        primary_tmp = st.session_state.tickers_selected[0]
        outlook_prompt = (
            f"Provide numeric forecasts for **EPS** and **Total Revenue** for {primary_tmp}'s "
            f"next quarter. Include your prediction, Street consensus, and beat probability in %. "
            f"Add one sentence of reasoning ending with 'Source: …'. "
            f"Return in markdown: a table plus bullets, no code fences."
        )
        with st.spinner("Contacting LLM…"):
            raw_md = ask_openai(
                model, "You are a senior equity analyst, precise and data-driven.",
                outlook_prompt)
        st.session_state.outlook_md = clean_llm_markdown(raw_md)
        st.experimental_rerun()
    else:
        st.markdown(f"<div class='card'>{st.session_state.outlook_md}</div>",
                    unsafe_allow_html=True)

# ───────────────────────── Ticker search UI ──────────────────────
st.markdown("### 📌 Stock Selection")
col1, col2 = st.columns([3, 2])

with col1:
    q = st.text_input("Search company or ticker", "", key="search_box")

with col2:
    current_list = st.session_state.tickers_selected
    if current_list:
        primary = st.selectbox("Primary ticker", options=current_list, index=0,
                               key="primary_select")
    else:
        primary = None

if len(q) >= 2:
    matches = search_ticker_symbols(q)
    if matches:
        display_opts = [f"{m['name']} ({m['symbol']})" for m in matches]
        choice = st.selectbox("Suggestions", display_opts, index=0, key="suggest_box")
        if st.button("➕ Add to basket", key="add_btn"):
            chosen_sym = choice.split("(")[-1].rstrip(")")
            default_seed = {"AAPL", "MSFT"}
            if set(st.session_state.tickers_selected) == default_seed:
                st.session_state.tickers_selected = []
            if chosen_sym not in st.session_state.tickers_selected:
                st.session_state.tickers_selected.insert(0, chosen_sym)
            st.rerun()
    else:
        st.info("No matches yet… keep typing")

tickers = st.session_state.tickers_selected
if not tickers:
    st.info("Add at least one ticker to proceed.")
    st.stop()

# ───────────────────────── Snapshot & metadata ───────────────────
summary = get_stock_summary(primary); add_to_history("bot", summary)
try:
    info = yf.Ticker(primary).info
    sector, industry = info.get("sector", ""), info.get("industry", "")
except Exception:
    info, sector, industry = {}, "", ""

try:
    hist = yf.Ticker(primary).history(period="5d")["Close"]
    last_px = hist.iloc[-1]
    pct_px = (last_px - hist.iloc[-2]) / hist.iloc[-2] * 100
except Exception:
    last_px = pct_px = float("nan")

with st.sidebar:
    st.markdown("### 🧾 Snapshot", unsafe_allow_html=True)
    st.markdown(f"""
        <div style='text-align:center;font-size:30px;font-weight:bold;color:white;'>
            ${last_px:.2f}
        </div>
        <div style='text-align:center;font-size:16px;color:{"green" if pct_px>=0 else "red"};'>
            {pct_px:+.2f}%
        </div>
        <hr style='margin:10px 0;border:1px solid #333;'/>
        <div style='font-size:13px;'>Market Cap: <b>${info.get('marketCap',0)/1e9:.2f} B</b></div>
        <div style='font-size:13px;'>P/E Ratio: <b>{info.get('trailingPE','—')}</b></div>
    """, unsafe_allow_html=True)

# ───────────────────────── Domain & competitors ──────────────────
domains = [d for d in (sector, industry) if d] or ["General"]
domain_selected = st.selectbox("Domain context", domains)

if len(tickers) == 1:
    competitors_all = fetch_competitors_llm(model, primary, domain_selected)
    basket = [primary] + competitors_all[:3]
else:
    competitors_all = tickers[1:]
    basket = tickers

# ───────────────────────── Strategy section (main) ───────────────
st.subheader("📑 Strategy Designer")
default_sector, default_avoid = sector or industry or "", primary
sector_in = st.text_input("Sector focus", default_sector)
goal      = st.selectbox("Positioning goal", ["Long", "Short", "Hedged", "Neutral"])
avoid_sym = st.text_input("Hedge / avoid ticker", default_avoid)
capital   = st.number_input("Capital (USD)", 1000, step=1000, value=10000)
horizon   = st.slider("Time horizon (months)", 1, 24, 6)

with st.expander("⚖️ Risk Controls", False):
    beta_rng  = st.slider("Beta match band", 0.5, 1.5, (0.8, 1.2), 0.05)
    stop_loss = st.slider("Stop-loss for shorts (%)", 1, 20, 10)

if st.button("Suggest Strategy"):
    basket_txt = ", ".join(basket)
    prompt = (
        f"Design a {goal.lower()} equity strategy using the basket [{basket_txt}]. "
        f"Sector focus: {sector_in}. Hedge or avoid exposure to {avoid_sym}. "
        f"Allocate a total of ${capital} over a {horizon}-month time horizon. "
        f"Match pair betas within the range {beta_rng[0]:.2f}-{beta_rng[1]:.2f}, "
        f"and apply a {stop_loss}% stop-loss to shorts. \n\n"
        "Return a markdown table with columns: Ticker, Position, Amount, Rationale, "
        "then a concise summary and 2-3 risk factors with explicit sources."
    )
    with st.spinner("Generating…"):
        plan = ask_openai(model, "You are a portfolio strategist.", prompt)

    st.markdown("### 📌 Suggested Strategy")
    st.write(plan)

    # Extract and highlight risks
    match = re.search(r"(### Risks.*?)(?=\n### |\Z)", plan, re.DOTALL | re.I)
    if match:
        st.markdown("### ⚠️ Highlighted Risks")
        st.markdown(
            f"<div class='card'><pre style='white-space:pre-wrap;font-size:14px;'>"
            f"{match.group(1).strip()}</pre></div>",
            unsafe_allow_html=True)
    else:
        st.info("No specific risks cited by the model.")

# ───────────────────────── Charts (conditional) ──────────────────
if show_charts:
    st.subheader("📈 Price Comparison")
    duration = st.selectbox("Duration", ["1mo", "3mo", "6mo", "1y"], 2, key="dur_sel")
    comps_selected = st.multiselect("Tickers to plot", options=basket + competitors_all,
                                    default=basket, key="plot_sel")
    if "SPY" not in comps_selected:
        comps_selected.append("SPY")
    price_df = fetch_stock_df(comps_selected, duration)
    if price_df.empty:
        st.error("No price data.")
    else:
        st.plotly_chart(px.line(price_df, title=f"Prices ({duration})",
                                labels={"value": "Price", "variable": "Ticker"}),
                        use_container_width=True)
        st.markdown("### 💹 Latest Prices")
        cols = st.columns(len(price_df.columns))
        for c, sym in zip(cols, price_df.columns):
            ser = price_df[sym]
            last = ser.iloc[-1]
            delta = ser.pct_change().iloc[-1] * 100
            with c:
                st.plotly_chart(
                    px.line(ser, height=80).update_layout(showlegend=False,
                                                          margin=dict(l=0,r=0,t=0,b=0),
                                                          xaxis=dict(showticklabels=False),
                                                          yaxis=dict(showticklabels=False)),
                    use_container_width=True)
                st.markdown(f"<div style='font-size:20px;font-weight:bold;'>{sym}</div>"
                            f"<div style='font-size:18px;'>${last:.2f}</div>"
                            f"<div style='color: {'green' if delta>=0 else 'red'};'>"
                            f"{delta:+.2f}%</div>", unsafe_allow_html=True)

# ───────────────────────── Optional Chat (kept) ──────────────────
st.markdown("---")
st.header("💬 Quick Chat (optional)")
for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)
q = st.chat_input("Ask anything…")
if q:
    add_to_history("user", q)
    ctx = f"Summary: {summary}\nDomain: {domain_selected}\nTickers: {', '.join(basket)}"
    ans = ask_openai(model, "You are a helpful market analyst.", ctx + "\n\n" + q)
    add_to_history("assistant", ans)
    st.rerun()
