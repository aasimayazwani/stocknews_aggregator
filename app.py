# app.py — Market-Movement Chatbot (portfolio-aware + risk-scan edition)
# ============================================================================
#  • Build a ticker portfolio via autocomplete search
#  • Edit / delete holdings and $ amounts in an editable table
#  • Scan near-term “headline risks” via GPT
#  • Generate size-aware strategy suggestions
#  • Optional price comparison chart + quick chat
# ============================================================================

from __future__ import annotations

# ── Standard library ─────────────────────────────────────────────────────────
import re, textwrap
from typing import List

# ── 3rd-party ────────────────────────────────────────────────────────────────
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# ── Local helpers (you already have these modules) ───────────────────────────
from config import DEFAULT_MODEL
from openai_client import ask_openai

# ────────────────────────────── PAGE SETUP ───────────────────────────────────
st.set_page_config(page_title="Strategy Chatbot", layout="wide")
st.title("🎯  Equity Strategy Assistant")

# ── Minimal dark-style CSS (cards, metrics, risk grid) ───────────────────────
st.markdown(
    """
    <style>
      .card{background:#1e1f24;padding:18px;border-radius:12px;margin-bottom:18px;}
      .metric{font-size:18px;font-weight:600;margin-bottom:2px;} .metric-small{font-size:14px;}
      .risk-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin:10px 0 16px;}
      .risk-card{background:#1f2937;border-radius:10px;padding:12px 16px;color:#f8fafc;box-shadow:0 0 0 1px #33415544;}
      .risk-card:hover{background:#273449;} .risk-card label{display:flex;justify-content:space-between;align-items:center;font-size:14px;font-weight:500;margin:0;}
      .risk-card input{margin-right:10px;transform:scale(1.2);accent-color:#10b981;} .risk-card a{margin-left:12px;color:#60a5fa;text-decoration:none;font-size:14px;}
      .risk-card a:hover{text-decoration:underline;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────────────── SESSION DEFAULTS ─────────────────────────────────
if "history"     not in st.session_state: st.session_state.history     = []
if "portfolio"   not in st.session_state: st.session_state.portfolio   = ["AAPL", "MSFT"]
if "alloc_df"    not in st.session_state: st.session_state.alloc_df    = pd.DataFrame()
if "risk_cache"  not in st.session_state: st.session_state.risk_cache  = {}
if "risk_ignore" not in st.session_state: st.session_state.risk_ignore = []
if "outlook_md"  not in st.session_state: st.session_state.outlook_md  = None

# ────────────────────────── HELPER FUNCTIONS ────────────────────────────────
def clean_md(md: str) -> str:
    """Remove formatting artifacts that sometimes leak from the LLM."""
    md = re.sub(r"(\d)(?=[A-Za-z])", r"\1 ", md)
    return md.replace("*", "").replace("_", "")

@st.cache_data(ttl=3600)
def search_tickers(query: str) -> List[str]:
    """Yahoo autocomplete → list of 'SYM – Name' strings."""
    try:
        r = requests.get(f"https://query1.finance.yahoo.com/v1/finance/search?q={query}", timeout=5)
        items = r.json().get("quotes", [])
        return [f"{x['symbol']} – {x.get('shortname', x.get('longname', ''))}" for x in items if "symbol" in x]
    except Exception:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers: List[str], period: str = "2d") -> pd.DataFrame:
    """2-day OHLC close → nicer tile display."""
    df = yf.download(tickers, period=period, progress=False)["Close"]
    return df.dropna(axis=1, how="all")

@st.cache_data(ttl=900, show_spinner=False)
def web_risk_scan(ticker: str, model_name: str = DEFAULT_MODEL) -> List[str]:
    """Ask GPT for five concise, current risk factors."""
    system = "You are a diligent equity risk analyst."
    user   = (f"Give *exactly* five short bullets (<20 words) listing near-term RISK FACTORS for {ticker}. "
              "Return as a Python list of strings.")
    raw = ask_openai(model_name, system_prompt=system, user_prompt=user)

    # Try literal-eval first; fallback to line-split
    try:
        import ast
        risks = [s.strip() for s in ast.literal_eval(raw.strip()) if isinstance(s, str)]
        return risks or [f"No clear near-term risks surfaced for {ticker}."]
    except Exception:
        lines = [ln.strip("•- ").strip() for ln in raw.splitlines() if ln.strip()]
        return lines[:5] or [f"No clear near-term risks surfaced for {ticker}."]

# ───────────────────────────── SIDEBAR ───────────────────────────────────────
with st.sidebar.expander("⚙️  Settings"):
    model = st.selectbox("OpenAI Model", [DEFAULT_MODEL, "gpt-4.1-mini", "gpt-4o-mini"], 0)
    if st.button("🧹  Clear chat history"): st.session_state.history = []
    if st.button("🗑️  Clear portfolio"):   st.session_state.portfolio = []
show_charts = st.sidebar.checkbox("📈  Show compar-chart", value=False)

# ────────────────── TICKER SEARCH + ADD BUTTON ──────────────────────────────
st.markdown("#### Add a stock/ETF to your portfolio")
query = st.text_input("🔎 Search ticker (symbol or name)", "")
if query:
    results = search_tickers(query)
    if results:
        choice = st.selectbox("Select result", results, key="ticker_select")
        sym = choice.split("–")[0].strip()
        if st.button("➕ Add", key="add_btn") and sym not in st.session_state.portfolio:
            st.session_state.portfolio.append(sym)
    else:
        st.info("No matches yet…")

# Short alias for readability throughout the script
portfolio = st.session_state.portfolio

# ─────────────── EDITABLE POSITION-SIZE TABLE (one source of truth) ─────────
st.markdown("### 💰 Position sizes")

# Initialize DataFrame if first run or portfolio changed
if st.session_state.alloc_df.empty:
    st.session_state.alloc_df = pd.DataFrame({"Ticker": portfolio, "Amount ($)": [10_000]*len(portfolio)})

# Remove rows whose ticker was deleted elsewhere; sort by size
st.session_state.alloc_df = (st.session_state.alloc_df
     .query("Ticker in @portfolio")
     .sort_values("Amount ($)", ascending=False, ignore_index=True)
)

# Editable grid (supports add/delete/edit)
alloc_df = st.data_editor(
    st.session_state.alloc_df,
    num_rows="dynamic",
    use_container_width=True,
    key="alloc_editor",
    hide_index=True,
)

# Clean edits → update session state objects
alloc_df = (alloc_df.dropna(subset=["Ticker"])
                      .query("Ticker != ''")
                      .drop_duplicates(subset=["Ticker"])
                      .sort_values("Amount ($)", ascending=False, ignore_index=True))

st.session_state.alloc_df        = alloc_df
st.session_state.portfolio       = alloc_df["Ticker"].tolist()
st.session_state.portfolio_alloc = dict(zip(alloc_df["Ticker"], alloc_df["Amount ($)"]))
portfolio = st.session_state.portfolio  # refresh alias

if not portfolio:
    st.error("Enter at least one ticker to continue.")
    st.stop()

# ───────────── 1-Day PRICE TILES ─────────────────────────────────────────────
tiles_df = fetch_prices(portfolio, "2d")
if not tiles_df.empty:
    last, prev = tiles_df.iloc[-1], tiles_df.iloc[-2]
    cols = st.columns(len(tiles_df.columns))
    for col, sym in zip(cols, tiles_df.columns):
        val   = last[sym]
        delta = (val - prev[sym]) / prev[sym] * 100
        with col:
            col.markdown(f"<div class='metric'>{sym}</div>", unsafe_allow_html=True)
            col.markdown(f"{val:,.2f}")
            col.markdown(f"<span class='metric-small' style='color:{'lime' if delta>=0 else 'tomato'}'>{delta:+.2f}%</span>",
                         unsafe_allow_html=True)

# ────────────── FOCUS STOCK + HEADLINE RISK GRID ────────────────────────────
primary = st.selectbox("🎯  Focus stock", portfolio, 0)
others  = [t for t in portfolio if t != primary]
basket  = [primary] + others

# Cache risk list by ticker
if primary not in st.session_state.risk_cache:
    with st.spinner("Scanning news with ChatGPT…"):
        st.session_state.risk_cache[primary] = web_risk_scan(primary, model)

risk_list = st.session_state.risk_cache[primary]

st.markdown("### 🔍 Key headline risks")
st.markdown("<div class='risk-grid'>", unsafe_allow_html=True)

selected_risks = []
for i, risk in enumerate(risk_list):
    key = f"risk_{i}"
    if key not in st.session_state: st.session_state[key] = True  # default checked
    checked = "checked" if st.session_state[key] else ""
    url = f"https://www.google.com/search?q={primary}+{risk.replace(' ', '+')}"  # dummy source

    st.markdown(
        f"""
        <div class='risk-card'>
          <label>
            <input type="checkbox" id="{key}" {checked}>
            <span>{risk}</span>
            <a href="{url}" target="_blank">ℹ️</a>
          </label>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state[key]:
        selected_risks.append(risk)

st.markdown("</div>", unsafe_allow_html=True)
st.session_state.risk_ignore = [r for r in risk_list if r not in selected_risks]

# ────────────────────────── STRATEGY DESIGNER ───────────────────────────────
st.markdown("### 📝 Strategy Designer")

sector_guess = yf.Ticker(primary).info.get("sector", "")
sector_in    = st.text_input("Sector", sector_guess)
goal         = st.selectbox("Positioning goal", ["Long", "Short", "Hedged", "Neutral"])
avoid_sym    = st.text_input("Hedge / avoid ticker", primary)
capital      = st.number_input("Capital (USD)", 1000, 1_000_000, 10_000, 1000)
horizon      = st.slider("Time horizon (months)", 1, 24, 6)

with st.expander("⚖️ Risk controls"):
    beta_rng  = st.slider("Beta match band", 0.5, 1.5, (0.8, 1.2), 0.05)
    stop_loss = st.slider("Stop-loss for shorts (%)", 1, 20, 10)

# Build prompt and call LLM
if st.button("Suggest strategy", type="primary"):
    alloc_str = "; ".join(f"{k}: ${v:,.0f}" for k, v in st.session_state.portfolio_alloc.items())
    ignored   = "; ".join(st.session_state.risk_ignore) or "None"
    risk_str  = ", ".join(risk_list)

    prompt = textwrap.dedent(f"""
        Act as a **portfolio strategist**.

        • **Basket**: {', '.join(basket)}
        • **Current allocation**: {alloc_str}
        • **Sector**: {sector_in}
        • **Goal**: {goal.lower()}
        • **Hedge / avoid**: {avoid_sym}
        • **Capital**: ${capital:,.0f}
        • **Horizon**: {horizon} months
        • **Beta band**: {beta_rng[0]:.2f}–{beta_rng[1]:.2f}
        • **Stop-loss**: {stop_loss} %
        • **Detected headline risks**: {risk_str}
        • **Ignore**: {ignored}

        **Return EXACTLY in this order**:
        1️⃣ Table — **Ticker | Position | Amount ($) | Rationale | Source**  
        2️⃣ `### Summary` (≤300 chars)  
        3️⃣ `### Residual Risks` (numbered; each ends with URL)
    """).strip()

    with st.spinner("Calling ChatGPT…"):
        raw_md = ask_openai(model, "You are a precise, citation-rich strategist.", prompt)

    plan_md = clean_md(raw_md)
    st.subheader("📌 Suggested strategy")
    st.markdown(plan_md, unsafe_allow_html=True)

    # Quick highlight of residual-risk section
    match = re.search(r"### Residual Risks.*", plan_md, flags=re.I | re.S)
    if match:
        st.subheader("⚠️ Residual Risks (quick view)")
        st.markdown(f"<div class='card'>{match.group(0)}</div>", unsafe_allow_html=True)

# ─────────────────────── OPTIONAL PRICE CHART ───────────────────────────────
if show_charts:
    st.markdown("### 📈 Price comparison")
    duration = st.selectbox("Duration", ["1mo", "3mo", "6mo", "1y"], 2)
    plot_tickers = st.multiselect("Tickers to plot", basket + ["SPY"], basket)
    if "SPY" not in plot_tickers: plot_tickers.append("SPY")
    df = fetch_prices(plot_tickers, duration)
    if df.empty:
        st.warning("No price data.")
    else:
        st.plotly_chart(
            px.line(df, title=f"Adjusted close ({duration})", labels={"value": "Price", "variable": "Ticker"}),
            use_container_width=True,
        )

# ───────────────────────── QUICK CHAT SECTION ───────────────────────────────
st.divider()
st.markdown("### 💬 Quick chat")

# Display chat history
for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

# Capture new user input
if user_q := st.chat_input("Ask anything…"):
    ctx = f"User portfolio: {', '.join(portfolio)}. Focus: {primary}."
    st.session_state.history.append(("user", user_q))
    response = ask_openai(model, "You are a helpful market analyst.", ctx + "\n\n" + user_q)
    st.session_state.history.append(("assistant", response))
    st.rerun()
