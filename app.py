# app.py – Market-Movement Chatbot  (portfolio-aware + risk-scan edition)
from __future__ import annotations

import re, textwrap, requests
from typing import List
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from config import DEFAULT_MODEL          # local module
from openai_client import ask_openai      # wrapper around OpenAI API
from stock_utils import get_stock_summary # your own helper
# ────────────────────────────────── THEME ─────────────────────────────────
st.set_page_config(page_title="Hedge Strategy Chatbot", layout="wide")

st.markdown(
    """
    <style>
      /* General card styling */
      .card {
        background: #1e1f24;
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 18px;
      }

      /* Ticker chip badge */
      .chip {
        display: inline-block;
        margin: 0 6px 6px 0;
        padding: 4px 10px;
        border-radius: 14px;
        background: #33415588;
        color: #f1f5f9;
        font-weight: 600;
        font-size: 13px;
      }

      /* Metrics (price % changes) */
      .metric {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 2px;
      }

      .metric-small {
        font-size: 14px;
      }

      /* Label style for form fields */
      label {
        font-weight: 600;
        font-size: 0.88rem;
      }

      /* Risk section grid layout */
      .risk-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 12px;
        margin-top: 10px;
        margin-bottom: 16px;
      }

      /* Individual risk card */
      .risk-card {
        background-color: #1f2937;
        border-radius: 10px;
        padding: 12px 16px;
        color: #f8fafc;
        box-shadow: 0 0 0 1px #33415544;
        transition: background 0.2s ease-in-out;
      }

      .risk-card:hover {
        background-color: #273449;
      }

      /* Checkbox label inside card */
      .risk-card label {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 14px;
        font-weight: 500;
        width: 100%;
        margin: 0;
        cursor: pointer;
      }

      .risk-card input[type="checkbox"] {
        margin-right: 10px;
        transform: scale(1.2);
        accent-color: #10b981; /* Tailwind green-500 */
      }

      /* Link icon inside card */
      .risk-card a {
        color: #60a5fa;
        text-decoration: none;
        font-size: 14px;
        margin-left: 12px;
      }

      .risk-card a:hover {
        text-decoration: underline;
      }

      /* Optional: icon if used */
      .risk-card i {
        font-style: normal;
        font-size: 13px;
        color: #60a5fa;
        margin-left: 6px;
      }
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


@st.cache_data(ttl=3600)
def search_tickers(query):
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"
    resp = requests.get(url)
    results = resp.json().get("quotes", [])
    return [f"{r['symbol']} – {r['shortname']}" for r in results if "shortname" in r]

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers: List[str], period="2d"):
    df = yf.download(tickers, period=period, progress=False)["Close"]
    return df.dropna(axis=1, how="all")

@st.cache_data(ttl=900, show_spinner=False)
# ─────────────────── RISK-SCAN via ChatGPT instead of DuckDuckGo ──────────────────
def web_risk_scan(ticker: str, model_name: str = DEFAULT_MODEL) -> List[str]:
    """
    Ask the LLM to return 4–5 hedge-relevant macro/sector risks affecting a given stock.
    Each item is concise and grounded in current economic or market context.
    """
    system = (
        "You are a portfolio hedging strategist. "
        "Your job is to identify macroeconomic, geopolitical, and sector-specific risk exposures "
        "that should be hedged against for large equity positions."
    )

    user = (
        f"What are the 4–5 most relevant macro or sector-level RISK FACTORS "
        f"that could affect {ticker}'s price or industry in the near term?\n\n"
        "Return the answer as a plain Python list of short strings (each under 20 words), "
        "e.g., ['Rising interest rates', 'Semiconductor supply chain issues', …].\n\n"
        "No explanation, just the list."
    )

    raw = ask_openai(model=model_name, system_prompt=system, user_prompt=user)

    # Try parsing the LLM reply as a Python list
    try:
        import ast

        lst = ast.literal_eval(raw.strip())
        risks = [s.strip() for s in lst if isinstance(s, str) and s.strip()]
        return risks or [f"No hedge-relevant risks identified for {ticker}."]
    except Exception:
        # Fallback: use line or comma splitting
        lines = [ln.strip("•- ").strip() for ln in raw.splitlines()]
        risks = [ln for ln in lines if ln]
        return risks[:5] or [f"No hedge-relevant risks identified for {ticker}."]

# ────────────────────────── SIDEBAR – SETTINGS ───────────────────────
with st.sidebar.expander("⚙️  Settings"):
    model = st.selectbox("OpenAI Model", [DEFAULT_MODEL, "gpt-4.1-mini", "gpt-4o-mini"], 0)
    if st.button("🧹  Clear chat history"):  st.session_state.history = []
    if st.button("🗑️  Clear portfolio"):    st.session_state.portfolio = []

show_charts = st.sidebar.checkbox("📈  Show compar-chart", value=False)

# ───────────────────────────── PORTFOLIO UI ──────────────────────────
# ⬇️ NEW ticker search & autocomplete with live API results
import requests

@st.cache_data(ttl=3600)
def search_tickers(query):
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"
    try:
        resp = requests.get(url, timeout=5)
        results = resp.json().get("quotes", [])
        return [f"{r['symbol']} – {r.get('shortname', r.get('longname', ''))}" for r in results if "symbol" in r]
    except Exception as e:
        return []

# Interactive ticker search section
st.markdown("#### Add a stock/ETF to your portfolio")
query = st.text_input("🔎 Search for ticker by name or symbol (e.g., 'Microsoft', 'AAPL')", "")

if query:
    options = search_tickers(query)
    if options:
        selected = st.selectbox("Choose from results", options, key="ticker_select")
        ticker_symbol = selected.split("–")[0].strip()
        if st.button("➕ Add to portfolio"):
            if ticker_symbol not in st.session_state.portfolio:
                st.session_state.portfolio.append(ticker_symbol)
    else:
        st.warning("No matching tickers found.")

# ─────────────────── 💰 POSITION-SIZE EDITOR ────────────────────
st.markdown("### 💰 Position sizes Editable")

# 1. Boot-strap a persistent table in session-state
if "alloc_df" not in st.session_state:
    tickers = st.session_state.portfolio
    st.session_state.alloc_df = pd.DataFrame({
        "Ticker": tickers,
        "Amount ($)": [10_000] * len(tickers)
    })

# 2. Keep only rows whose ticker is still in session_state.portfolio
st.session_state.alloc_df = (
    st.session_state.alloc_df
      .query("Ticker in @st.session_state.portfolio")          # auto-drop removed tickers
      .sort_values("Amount ($)", ascending=False, ignore_index=True)
)

# 3. Show an editable, sortable table   (Streamlit ≥1.29)
editor_df = st.data_editor(
    st.session_state.alloc_df,
    num_rows="dynamic",
    use_container_width=True,
    key="alloc_editor",        # preserves edits between reruns
    hide_index=True,
)

# 4. Persist edits AND detect deletions
clean_df = (
    editor_df
      .dropna(subset=["Ticker"])                # empty row = deleted
      .query("Ticker != ''")
      .drop_duplicates(subset=["Ticker"])       # guard against accidental dups
      .sort_values("Amount ($)", ascending=False, ignore_index=True)
)

st.session_state.alloc_df       = clean_df                      # overwrite canonical table
st.session_state.portfolio      = clean_df["Ticker"].tolist()   # keep master list in sync
st.session_state.portfolio_alloc = dict(
    zip(clean_df["Ticker"], clean_df["Amount ($)"])
)


portfolio = st.session_state.portfolio

# Chips + price tiles
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

# ────────────────────── headline-risk retrieval (cached) ─────────────────────
if primary not in st.session_state.risk_cache:
    with st.spinner("Scanning news with ChatGPT…"):
        st.session_state.risk_cache[primary] = web_risk_scan(primary, model)

# ────────────────────── AUTOMATED RISK SCAN SECTION ───────────────────
st.markdown("### 🔍  Key headline risks")

if primary not in st.session_state.risk_cache:
    with st.spinner("Scanning web…"):
        st.session_state.risk_cache[primary] = web_risk_scan(primary)

risk_list = st.session_state.risk_cache[primary]
# Dummy mapping of risk → URL (replace with real scraping or LLM output if available)
risk_links = {
    r: f"https://www.google.com/search?q={primary}+{r.replace(' ', '+')}" for r in risk_list
}

st.markdown("Un-check any headline you **do not** want the LLM to consider:")

# --------------------------------------------
# 🧠 2-Column Responsive Risk Rendering Section
# --------------------------------------------
selected_risks = []

# Generate dummy source links if needed
risk_links = {
    r: f"https://www.google.com/search?q={primary}+{r.replace(' ', '+')}" for r in risk_list
}

# Begin the grid container
st.markdown("<div class='risk-grid'>", unsafe_allow_html=True)

# Render each risk in a styled card with checkbox + source
for i, risk in enumerate(risk_list):
    key = f"risk_{i}"
    # Default all to checked unless already set
    if key not in st.session_state:
        st.session_state[key] = True

    # Maintain checked state for each
    checked_attr = "checked" if st.session_state[key] else ""

    html = f"""
    <div class='risk-card'>
      <label for="{key}">
        <input type="checkbox" id="{key}" name="{key}" onclick="window.dispatchEvent(new Event('input'))" {checked_attr}>
        <span>{risk}</span>
        <a href="{risk_links[risk]}" target="_blank">ℹ️</a>
      </label>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

    # Track which are still selected
    if st.session_state[key]:
        selected_risks.append(risk)

# End the grid
st.markdown("</div>", unsafe_allow_html=True)

# Update the exclusion list in session state
st.session_state.risk_ignore = [r for r in risk_list if r not in selected_risks]


#st.session_state.risk_ignore = [r for r in risk_list if r not in exclude]

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
        md = ask_openai(model, "You are a senior hedge fund analyst.", p)
        st.session_state.outlook_md = clean_md(md); st.rerun()
    else:
        st.markdown(f"<div class='card'>{st.session_state.outlook_md}</div>", unsafe_allow_html=True)

# ─────────────────────────── STRATEGY DESIGNER ───────────────────────
st.markdown("### 📝  Strategy Designer")
sector_guess = yf.Ticker(primary).info.get("sector", "")
sector_in    = st.text_input("Sector", sector_guess)
#goal         = st.selectbox("Positioning goal", ["Long", "Short", "Hedged", "Neutral"])
#avoid_sym    = st.text_input("Hedge / avoid ticker", primary)
capital      = st.number_input("Capital (USD)", 1000, 1_000_000, 10_000, 1000)
horizon      = st.slider("Time horizon (months)", 1, 24, 6)

with st.expander("⚖️  Risk controls"):
    beta_rng  = st.slider("Beta match band", 0.5, 1.5, (0.8, 1.2), 0.05)
    stop_loss = st.slider("Stop-loss for shorts (%)", 1, 20, 10)

# ─────────────────────── Strategy generation & rendering ───────────────────────
if st.button("Suggest strategy", type="primary"):
    # 1.  Build prompt ----------------------------------------------------------
    ignored = "; ".join(st.session_state.risk_ignore) or "None"
    risk_string = ", ".join(risk_list) or "None"
    alloc_str = "; ".join(
        f"{k}: ${v:,.0f}" for k, v in st.session_state.portfolio_alloc.items()
    ) or "None provided"
    prompt = textwrap.dedent(f"""
        Act as a **hedging strategist**.

        • **Basket**: {', '.join(basket)}
        • **Current allocation**: {alloc_str}
        • **Total capital** (based on portfolio table): ${total_capital:,.0f}
        • **Horizon**: {horizon} months
        • **Beta band**: {beta_rng[0]:.2f}–{beta_rng[1]:.2f}
        • **Stop-loss**: {stop_loss} %
        • **Detected headline risks** for {primary}: {risk_string}
        • **Ignore** the following risks when constructing the hedge: {ignored}

        Your task is to design a tactical hedge that offsets risk while preserving exposure to high-conviction positions.

        **Return EXACTLY in this markdown order**:

        1️⃣ A table (markdown pipe format) with columns **Ticker | Position | Amount ($) | Rationale | Source**  
        – Put the full clickable URL in the *Source* column of each row.

        2️⃣ `### Summary` – 2–3 plain sentences (max 300 chars) – NO italics/bold inside.

        3️⃣ `### Residual Risks` – a numbered list; each bullet ≤ 25 words and **MUST end with a source URL in parentheses**.

        Do not wrap anything in code-fences.
    """).strip()


    # 2.  Call OpenAI -----------------------------------------------------------
    with st.spinner("Calling ChatGPT…"):
        raw_md = ask_openai(model, "You are a precise, citation-rich strategist.", prompt)

    # 3.  Clean & show ----------------------------------------------------------
    plan_md = clean_md(raw_md)
    st.subheader("📌 Suggested strategy")
    st.markdown(plan_md, unsafe_allow_html=True)

    # 4.  Optionally pull out Residual Risks (to highlight in a card) ----------
    match = re.search(r"### Residual Risks.*", plan_md, flags=re.I | re.S)
    if match:
        st.subheader("⚠️ Residual Risks (quick view)")
        st.markdown(f"<div class='card'>{match.group(0)}</div>", unsafe_allow_html=True)

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
