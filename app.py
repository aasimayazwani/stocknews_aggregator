from __future__ import annotations
import re, textwrap, requests
from typing import List
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
from config import DEFAULT_MODEL          # local module
from openai_client import ask_openai      # wrapper around OpenAI API
from stock_utils import get_stock_summary # your own helper

# ────────────────────────────────── THEME ─────────────────────────────────
st.set_page_config(page_title="Hedge Strategy Chatbot", layout="centered")

with st.sidebar.expander("📌 Investor Profile", expanded=False):
    # Experience
    experience = st.selectbox(
        label="",
        options=["Beginner", "Intermediate", "Expert"],
        index=["Beginner", "Intermediate", "Expert"].index(st.session_state.get("experience_level", "Expert")),
        format_func=lambda x: f"Experience: {x}",
        key="experience_level"
    )

    # Detail level
    detail_level = st.selectbox(
        label="",
        options=["Just the strategy", "Explain the reasoning", "Both"],
        index=["Just the strategy", "Explain the reasoning", "Both"].index(st.session_state.get("explanation_pref", "Just the strategy")),
        format_func=lambda x: f"Detail level: {x}",
        key="explanation_pref"
    )

    # Time horizon
    horizon = st.slider(
        label="Time horizon (months):",
        min_value=1,
        max_value=24,
        value=st.session_state.get("time_horizon", 6),
        key="time_horizon"
    )

    # ── Experience-based default instruments ──
    experience_defaults = {
        "Beginner": ["Inverse ETFs", "Commodities"],
        "Intermediate": ["Put Options", "Inverse ETFs", "Commodities"],
        "Expert": [
            "Put Options", "Collar Strategy", "Inverse ETFs", "Short Selling",
            "Volatility Hedges", "Commodities", "FX Hedges"
        ]
    }

    all_options = [
        "Put Options", "Collar Strategy", "Inverse ETFs", "Short Selling",
        "Volatility Hedges", "Commodities", "FX Hedges"
    ]

    current_exp = st.session_state.get("experience_level", "Beginner")

    if "prev_experience" not in st.session_state or st.session_state.prev_experience != current_exp:
        st.session_state.allowed_instruments = experience_defaults.get(current_exp, [])
        st.session_state.prev_experience = current_exp

    st.multiselect(
        "Allowed hedge instruments:",
        options=all_options,
        default=st.session_state.allowed_instruments,
        key="allowed_instruments"
    )

with st.sidebar.expander("🧮 Investment Settings", expanded=True):
    pass  # Removed Focus stock selectbox

with st.sidebar.expander("⚙️ Strategy Settings", expanded=False):
    st.slider("🎯 Beta match band", 0.5, 2.0, (1.15, 1.50), step=0.01, key="beta_band")
    st.slider("🔻 Stop-loss for shorts (%)", 1, 20, 10, key="stop_loss")
    st.slider("💰 Total hedge budget (% of capital)", 5, 25, 10, key="total_budget")
    st.slider("📉 Max per single hedge (% of capital)", 1, 10, 5, key="max_hedge")

with st.sidebar.expander("🧹 Session Tools", expanded=False):
    with st.sidebar.expander("🧠 Previous Strategies", expanded=True):
        history = st.session_state.get("strategy_history", [])

        if not history:
            st.info("No previous strategies yet.")
        else:
            for idx, run in reversed(list(enumerate(history))):
                with st.expander(f"Run {idx+1} — {run['timestamp']} | Horizon: {run['horizon']} mo"):
                    st.markdown(
                        f"**Capital**: ${run['capital']:,.0f}  \n"
                        f"**Beta Band**: {run['beta_band'][0]}–{run['beta_band'][1]}"
                    )
                    st.dataframe(run["strategy_df"], use_container_width=True)
                    st.markdown("**Strategy Rationale**")
                    st.markdown(run["rationale_md"])

    suggest_clicked = st.sidebar.button("🚀 Suggest strategy", type="primary", use_container_width=True)
    if st.button("🗑️ Clear Portfolio"):
        st.session_state.portfolio_alloc = {}
    if st.button("🧽 Clear Chat History"):
        st.session_state.chat_history = []
    if st.button("🗑️ Clear Strategy History"):
        st.session_state.strategy_history = []

# 🔧 Extract sidebar values into variables
experience_level   = st.session_state.get("experience_level", "Expert")
explanation_pref   = st.session_state.get("explanation_pref", "Just the strategy")
avoid_overlap      = st.session_state.get("avoid_overlap", True)
allowed_instruments = st.session_state.get("allowed_instruments", ["Put Options", "Collar Strategy"])
beta_rng           = st.session_state.get("beta_band", (1.15, 1.50))
stop_loss          = st.session_state.get("stop_loss", 10)
hedge_budget_pct   = st.session_state.get("total_budget", 10)
single_hedge_pct   = st.session_state.get("max_hedge", 5)
horizon            = st.session_state.get("time_horizon", 6)
portfolio          = st.session_state.get("portfolio", ["AAPL", "MSFT"])  # Use all portfolio stocks

st.markdown("""
<style>
  /* 🎨 Select and Multiselect Styling */
  div[data-baseweb="select"] > div {
    background-color: #1f2937 !important;
    border-radius: 12px !important;
    padding: 4px 12px !important;
    color: #f1f5f9 !important;
    font-size: 15px !important;
    min-height: 40px !important;
    line-height: 1.4 !important;
    display: flex;
    align-items: center;
  }

  .stMultiSelect > div {
    gap: 4px !important;
    flex-wrap: wrap;
    padding: 2px !important;
  }

  .stMultiSelect span[data-baseweb="tag"] {
    margin-bottom: 4px !important;
    font-size: 13px !important;
    padding: 4px 8px !important;
    border-radius: 12px !important;
  }

  /* 🧱 Card Component */
  .card {
    background: #1e1f24;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 18px;
  }

  /* 🔷 Ticker Chip Badge */
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

  /* 📊 Metric Display */
  .metric {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 2px;
  }

  .metric-small {
    font-size: 14px;
  }

  /* 🏷️ Label Styling */
  label {
    font-weight: 600;
    font-size: 0.88rem;
  }

  /* ⚠️ Risk Section Grid */
  .risk-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 12px;
    margin-top: 10px;
    margin-bottom: 16px;
  }

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

  .risk-card label {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 14px;
    font-weight: 500;
    margin: 0;
    cursor: pointer;
  }

  .risk-card input[type="checkbox"] {
    margin-right: 10px;
    transform: scale(1.2);
    accent-color: #10b981;
  }

  .risk-card a {
    color: #60a5fa;
    text-decoration: none;
    font-size: 14px;
    margin-left: 12px;
  }

  .risk-card a:hover {
    text-decoration: underline;
  }

  .risk-card i {
    font-style: normal;
    font-size: 13px;
    color: #60a5fa;
    margin-left: 6px;
  }

  /* 🧼 Misc. Clean-up */
  h3 {
    margin-top: 0;
    margin-bottom: 0;
  }
</style>
""", unsafe_allow_html=True)

st.title("Equity Strategy Assistant")

# ─────────────────────────────── STATE ────────────────────────────────
if "history"     not in st.session_state: st.session_state.history     = []
if "portfolio"   not in st.session_state: st.session_state.portfolio   = ["AAPL", "MSFT"]
if "outlook_md"  not in st.session_state: st.session_state.outlook_md  = None
if "risk_cache"  not in st.session_state: st.session_state.risk_cache  = {}  # {ticker: [risks]}
if "risk_ignore" not in st.session_state: st.session_state.risk_ignore = []  # selected exclusions
if "selected_risks" not in st.session_state: st.session_state.selected_risks = []

# ──────────────────────────── HELPERS ────────────────────────────────
def clean_md(md: str) -> str:
    md = re.sub(r"(\d)(?=[A-Za-z])", r"\1 ", md)
    return md.replace("*", "").replace("_", "")

def render_rationale(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No hedge rationale to show.")
        return

    total = df["Amount ($)"].sum()
    st.markdown(
        f"A total of **${total:,.0f}** was allocated to hedge instruments "
        "to mitigate key risks in the portfolio.\n\n"
        "Below is the explanation for each hedge component:"
    )

    for _, row in df.iterrows():
        tick   = row.get("Ticker", "—").strip()
        pos    = row.get("Position", "—").title()
        amt    = row.get("Amount ($)", 0)
        rat    = row.get("Rationale", "No rationale provided").strip()
        src    = row.get("Source", "").strip()

        card  = (
            f"<div style='background:#1e293b;padding:12px;border-radius:10px;"
            f"margin-bottom:10px;color:#f1f5f9'>"
            f"<b>{tick} ({pos})</b> — "
            f"<span style='color:#22d3ee'>${amt:,.0f}</span><br>{rat}"
        )

        if re.match(r'^https?://', src):
            card += f"<br><a href='{src}' target='_blank' style='color:#60a5fa;'>Source ↗</a>"

        card += "</div>"
        st.markdown(card, unsafe_allow_html=True)

def fallback_ticker_lookup(name: str, model_name: str = "gpt-4.1-mini") -> str:
    prompt = f"What is the stock ticker symbol for the publicly traded company '{name}'?"
    raw = ask_openai(
        model=model_name,
        system_prompt="You are a financial assistant that returns only the correct stock ticker symbol.",
        user_prompt=prompt,
    )
    match = re.search(r"\b[A-Z]{2,5}\b", raw.strip())
    return match.group(0) if match else ""

@st.cache_data(ttl=3600)
def search_tickers(query: str) -> List[str]:
    from urllib.parse import quote
    query_clean = query.strip().lower()
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={quote(query_clean)}"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return []
        results = resp.json().get("quotes", [])
        tickers = [f"{r.get('symbol', '')} – {r.get('shortname') or r.get('longname') or ''}" for r in results if r.get('symbol') and (r.get('shortname') or r.get('longname'))]
        if not tickers and len(query_clean) >= 3:
            fallback = fallback_ticker_lookup(query_clean)
            if fallback:
                tickers.append(f"{fallback} – (GPT suggested)")
        return tickers
    except Exception:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers: List[str], period="2d"):
    df = yf.download(tickers, period=period, progress=False)["Close"]
    return df.dropna(axis=1, how="all")

@st.cache_data(ttl=900, show_spinner=False)
def web_risk_scan(ticker: str):
    api_key = st.secrets.get("NEWSAPI_KEY") or os.getenv("NEWSAPI_KEY")
    if not api_key:
        return [("⚠️ No NEWSAPI_KEY found. Please add it to .streamlit/secrets.toml", "#")]
    query = f'"{ticker}" AND (analyst OR downgrade OR rating OR earnings OR revise OR cut OR risk)'
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "language": "en", "sortBy": "publishedAt", "pageSize": 15, "apiKey": api_key}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        articles = response.json().get("articles", [])
    except Exception as e:
        return [(f"❌ Error fetching headlines: {str(e)}", "#")]
    known_analysts = ["Dan Ives", "Katy Huberty", "Gene Munster", "Mark Mahaney"]
    seen_titles = set()
    risks = []
    for article in articles:
        title = article.get("title", "")
        url = article.get("url", "#")
        if not title or title in seen_titles:
            continue
        if any(keyword in title.lower() for keyword in ["downgrade", "risk", "cut", "concern", "slashed", "fall", "caution", "bearish", "revised"]):
            risks.append((title, url))
            seen_titles.add(title)
        elif any(analyst in title for analyst in known_analysts):
            risks.append((f"📊 Analyst: {title}", url))
            seen_titles.add(title)
        if len(risks) >= 5:
            break
    if not risks:
        return [("No relevant analyst headlines found for " + ticker, "#")]
    return risks

# ─────────────────────────────── STATE ────────────────────────────────
if "history"     not in st.session_state: st.session_state.history     = []
if "portfolio"   not in st.session_state: st.session_state.portfolio   = ["AAPL", "MSFT"]
if "outlook_md"  not in st.session_state: st.session_state.outlook_md  = None
if "risk_cache"  not in st.session_state: st.session_state.risk_cache  = {}
if "risk_ignore" not in st.session_state: st.session_state.risk_ignore = []

# ────────────────────────── SIDEBAR – SETTINGS ───────────────────────
experience_to_default = {
    "Beginner": ["Inverse ETFs", "Commodities"],
    "Intermediate": ["Put Options", "Inverse ETFs", "Commodities"],
    "Expert": ["Put Options", "Collar Strategy", "Inverse ETFs", "Short Selling", "Volatility Hedges", "Commodities", "FX Hedges"]
}

avoid_duplicate_hedges = st.session_state.get("avoid_overlap", True)
st.session_state.avoid_dup_hedges = avoid_duplicate_hedges

# 🎯 Use all portfolio stocks for hedging
portfolio_stocks = st.session_state.portfolio

# ───────────────────────────── PORTFOLIO UI ──────────────────────────
st.markdown("### Position sizes Editable")

if "alloc_df" not in st.session_state:
    tickers = st.session_state.portfolio
    st.session_state.alloc_df = pd.DataFrame({
        "Ticker": tickers,
        "Amount ($)": [10_000] * len(tickers),
        "Stop-Loss ($)": [None] * len(tickers)
    })

st.session_state.alloc_df = (
    st.session_state.alloc_df
      .query("Ticker in @st.session_state.portfolio")
      .sort_values("Amount ($)", ascending=False, ignore_index=True)
)

clean_df = (
    st.session_state.alloc_df
      .dropna(subset=["Ticker"])
      .query("Ticker != ''")
      .drop_duplicates(subset=["Ticker"])
      .sort_values("Amount ($)", ascending=False, ignore_index=True)
)

tickers = clean_df["Ticker"].tolist()
prices_df = fetch_prices(tickers, period="2d")

if not prices_df.empty:
    last = prices_df.iloc[-1]
    prev = prices_df.iloc[-2]
    clean_df["Price"] = last.reindex(tickers).round(2).values
    clean_df["Δ 1d %"] = ((last - prev) / prev * 100).reindex(tickers).round(2).values
else:
    clean_df["Price"] = 0.0
    clean_df["Δ 1d %"] = 0.0

editor_df = st.data_editor(
    clean_df,
    disabled={"Price": True, "Δ 1d %": True},
    num_rows="dynamic",
    use_container_width=True,
    key="alloc_editor",
    hide_index=True,
)

st.session_state.stop_loss_map = dict(
    zip(editor_df["Ticker"], editor_df["Stop-Loss ($)"])
)
st.session_state.alloc_df = editor_df
st.session_state.portfolio = editor_df["Ticker"].tolist()
st.session_state.portfolio_alloc = dict(
    zip(editor_df["Ticker"], editor_df["Amount ($)"])
)

ticker_df = pd.DataFrame({
    "Ticker": list(st.session_state.portfolio_alloc.keys()),
    "Amount": list(st.session_state.portfolio_alloc.values())
}).sort_values("Amount", ascending=False)

ticker_df["Amount"] = ticker_df["Amount"].fillna(0)
ticker_df["Label"] = (
    ticker_df["Ticker"] + " ($" +
    ticker_df["Amount"].round(0).astype(int).astype(str) + ")"
)

portfolio = st.session_state.portfolio

with st.sidebar.expander("🔍 Key headline risks", expanded=True):
    for ticker in portfolio:
        if ticker not in st.session_state.risk_cache:
            with st.spinner(f"Scanning web for {ticker}…"):
                st.session_state.risk_cache[ticker] = web_risk_scan(ticker)

        risk_tuples = st.session_state.risk_cache[ticker]
        risk_titles = [t[0] for t in risk_tuples]
        risk_links = {title: url for title, url in risk_tuples}

        st.markdown(f"### Risks for {ticker}")
        selected_risks = []

        for i, risk in enumerate(risk_titles):
            key = f"risk_{ticker}_{i}"
            default = True if key not in st.session_state else st.session_state[key]

            cols = st.columns([0.1, 0.8, 0.1])
            with cols[0]:
                is_selected = st.checkbox("", key=key, value=default)
            with cols[1]:
                st.markdown(risk)
            with cols[2]:
                st.markdown(f"[ℹ️]({risk_links.get(risk, '#')})")

            if is_selected:
                selected_risks.append(risk)

        st.session_state.selected_risks = selected_risks
        st.session_state.risk_ignore = [r for r in risk_titles if r not in selected_risks]

experience_to_default = {
    "Beginner": ["Inverse ETFs", "Commodities"],
    "Intermediate": ["Put Options", "Inverse ETFs", "Commodities"],
    "Expert": ["Put Options", "Collar Strategy", "Inverse ETFs", "Short Selling", "Volatility Hedges", "Commodities", "FX Hedges"]
}

default_instruments = experience_to_default.get(st.session_state.experience_level, [])

if "strategy_history" not in st.session_state:
    st.session_state.strategy_history = []

if suggest_clicked:
    ignored = "; ".join(st.session_state.risk_ignore) or "None"
    total_capital = sum(st.session_state.portfolio_alloc.values())
    risk_string = ", ".join(r for ticker in portfolio for r in st.session_state.risk_cache.get(ticker, [])[0]) or "None"
    alloc_str = "; ".join(f"{k}: ${v:,.0f}" for k, v in st.session_state.portfolio_alloc.items()) or "None"

    exp_pref = st.session_state.explanation_pref

    experience_note = {
        "Beginner":     "Use plain language and define jargon the first time you use it.",
        "Intermediate": "Assume working knowledge of finance; keep explanations concise.",
        "Expert":       "Write in professional sell-side style; no hand-holding.",
    }[st.session_state.experience_level]

    if exp_pref == "Just the strategy":
        rationale_rule = "Each *Rationale* must be **≤ 25 words (one-two sentence)**."
    elif exp_pref == "Explain the reasoning":
        rationale_rule = ("Each *Rationale* must be **2 sentences totalling ≈ 30-60 words** "
                          "(logic + risk linkage).")
    else:  # "Both"
        rationale_rule = ("Each *Rationale* must be **3 sentences totalling ≈ 60-90 words** – "
                          "1️⃣ logic, 2️⃣ quantitative context, 3️⃣ trade-offs.")

    stop_loss_str = "; ".join(
        f"{ticker}: ${float(sl):.2f}" for ticker, sl in st.session_state.stop_loss_map.items() if pd.notnull(sl)
    ) or "None"

    hedge_budget_pct   = st.session_state.get("total_budget", 10)
    single_hedge_pct   = st.session_state.get("max_hedge", 5)
    max_hedge_notional = total_capital * hedge_budget_pct / 100

    avoid_note = ""
    if st.session_state.avoid_dup_hedges:
        avoid_note = (
            "- ❌ **Do NOT suggest hedge instruments already in the user’s portfolio** "
            f"({', '.join(st.session_state.portfolio)}).\n"
            "- ✅ Prefer diversifiers (sector ETFs, index futures, inverse ETFs, FX, commodities).\n"
        )

    prompt = textwrap.dedent(f"""
    👋  You are a **Hedging Strategist** helping investors protect capital while keeping portfolio beta between **{beta_rng[0]:.2f} – {beta_rng[1]:.2f}**.

    ─────────────────────────────────
    🔑 **Key Instructions**

    1. **Hedging Scope**  
    • Hedge all stocks in the portfolio: **{', '.join(portfolio)}**.

    2. **Allowed Hedge Types**  
    {', '.join(st.session_state.allowed_instruments)}  
    *Use only these. Do **NOT** propose anything else.*

    3. **Budget & Sizing Rules**  
    • Total hedge cost ≤ **{hedge_budget_pct}%** of capital (${total_capital:,.0f})  
    • Any single hedge ≤ **{single_hedge_pct}%** of capital  
    • Option premium target ≤ **3 %** of notional  
    • Max 5 hedges

    4. **When to Hedge**  
    Flag a position if:  
    • Its stop-loss sits ≥ 5 % above market **or**  
    • It shows high headline-risk sensitivity: {risk_string}  
    Ignore: {ignored}

    ─────────────────────────────────
    📊 **Portfolio Snapshot**

    • Positions: {', '.join(portfolio)}  
    • Allocation: {alloc_str}  
    • Stop-losses: {stop_loss_str or 'None'}  
    • Horizon: **{horizon} mo** • Capital: **${total_capital:,.0f}**

    ─────────────────────────────────
    📝 **Deliverables (Markdown only)**

    **A. Hedge List** – one bullet per idea, end each with reference tag `[n]`  
    `1. **AAPL** — Put. Buy 3× Aug 175P (≤ {rationale_rule}) [1]`

    **B. Sizing Table** – immediately after bullets  
    ```

| Hedge                 | Qty / Cts |                   \$ Notional |               % Capital |
| --------------------- | --------- | ----------------------------: | ----------------------: |
| AAPL Aug 175 P (puts) | 3         |                         3 000 |                     3 % |
| …                     | …         |                             … |                       … |
| **Total**             |           | ≤ {max_hedge_notional:,.0f} | ≤ {hedge_budget_pct}% |

    ```

    **C. References**  
    `[1] https://source.example`

    **D. Summary** – ≤ 300 chars

    **E. Residual Risks** – numbered list, ≤ 25 words each, each ending in a URL

    ─────────────────────────────────
    👤 **Investor Profile**

    Experience: **{st.session_state.experience_level}**  
    Detail level: **{st.session_state.explanation_pref}**

    Return the answer in plain Markdown, no HTML or code fences.
    """).strip()

    with st.spinner("Calling ChatGPT…"):
        raw_md = ask_openai(model, "You are a precise, citation-rich strategist.", prompt)

    plan_md = raw_md

    footnotes = dict(re.findall(r"\[(\d+)\]\s+(https?://[^\s]+)", plan_md))
    superscripts_map = "⁰¹²³⁴⁵⁶⁷⁸⁹"
    for ref_id, url in footnotes.items():
        superscript = "".join(superscripts_map[int(d)] for d in ref_id)
        plan_md = plan_md.replace(f"[{ref_id}]", f"[🔗{superscript}]({url})")
    plan_md = re.sub(r"^\[\d+\]\s+https?://[^\s]+", "", plan_md, flags=re.MULTILINE)

    lines = plan_md.splitlines()
    hedge_lines = [line for line in lines if re.match(r"^\d+\.\s+(?:\*\*)?(.+?)(?:\*\*)?\s+—\s+", line)]

    st.subheader("📌 Suggested strategy")

    records = []
    for line in hedge_lines:
        match = re.match(r"^(\d+)\.\s+(?:\*\*)?(.+?)(?:\*\*)?\s+—\s+(.*?)\s+\[(\d+)\]", line)
        if match:
            _, ticker, rationale, ref_id = match.groups()
            url = footnotes.get(ref_id, "#")
            records.append({
                "Ticker": ticker.strip(),
                "Rationale": rationale.strip(),
                "Source": url,
                "Position": "Hedge",
                "Amount ($)": 0
            })

    df = pd.DataFrame(records)

    user_df = editor_df.copy()
    user_df["Position"] = "Long"
    user_df["Source"] = "User portfolio"
    user_df["% of Portfolio"] = (user_df["Amount ($)"] / user_df["Amount ($)"].sum() * 100).round(2)
    user_df["Rationale"] = "—"
    user_df["Ticker"] = user_df["Ticker"].astype(str)

    df["% of Portfolio"] = 0
    df["Price"] = "_n/a_"
    df["Δ 1d %"] = "_n/a_"
    user_df["Price"] = user_df["Price"].round(2)
    user_df["Δ 1d %"] = user_df["Δ 1d %"].round(2)

    final_cols = ["Ticker", "Position", "Amount ($)", "% of Portfolio", "Price", "Δ 1d %", "Source", "Rationale"]
    user_df = user_df[final_cols]
    missing_cols = [col for col in final_cols if col not in df.columns]
    for col in missing_cols:
        df[col] = ""
    df = df[final_cols]
    combined_df = pd.concat([user_df, df], ignore_index=True)

    st.session_state.user_df = user_df
    st.session_state.strategy_df = df
    st.session_state.rationale_md = plan_md
    st.session_state.strategy_history.append({
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "horizon": horizon,
        "beta_band": beta_rng,
        "capital": total_capital,
        "strategy_df": df,
        "rationale_md": plan_md,
    })

    st.dataframe(combined_df.drop(columns=["Rationale"]), use_container_width=True)

    st.markdown("### 📌 Hedge Strategy Rationale")
    st.markdown(plan_md)

st.divider()
st.markdown("### 💬  Quick chat")
for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

if q := st.chat_input("Ask anything…"):
    ctx = f"User portfolio: {', '.join(portfolio)}. Focus: All stocks."
    st.session_state.history.append(("user", q))
    ans = ask_openai(model, "You are a helpful market analyst.", ctx + "\n\n" + q)
    st.session_state.history.append(("assistant", ans))
    st.rerun()