from __future__ import annotations
import re, textwrap, requests
from typing import List
import os
import streamlit as st
import pandas as pd
import json
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
    pass  # Removed Focus stock and other settings

with st.sidebar.expander("⚙️ Strategy Settings", expanded=False):
    pass  # Removed sliders for beta band, stop-loss, total budget, and max hedge

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
horizon            = st.session_state.get("time_horizon", 6)
portfolio          = st.session_state.get("portfolio", ["AAPL", "MSFT"])  # Use all portfolio stocks
model = DEFAULT_MODEL  # Define model variable using the imported DEFAULT_MODEL

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
if "risk_ignore" not in st.session_state: st.session_state.risk_ignore = []

# ─── after first STATE block ───
if "chosen_strategy" not in st.session_state:
    st.session_state.chosen_strategy = None
if "strategy_history" not in st.session_state:
    st.session_state.strategy_history = []

"""# ─── inside suggest_clicked, just after render_strategy_cards(df_strat) ───
st.session_state.strategy_history.append({
    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "horizon": horizon,
    "capital": total_capital,
    "strategy_df": df_strat,
})"""

# ──────────────────────────── HELPERS ────────────────────────────────
def render_strategy_cards(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("LLM returned no strategies.")
        return

    for i, row in df.iterrows():
        is_selected = (
            st.session_state.chosen_strategy
            and row.name == st.session_state.chosen_strategy.get("name")
        )

        border_color = "#10b981" if is_selected else "#334155"
        card = st.container()
        with card:
            st.markdown(
                f"<div style='border:1px solid {border_color}; border-radius:10px; padding:12px;'>",
                unsafe_allow_html=True
            )
            cols = st.columns([6, 2, 2])
            cols[0].markdown(f"### {row.name}")
            cols[1].metric("Risk ↓", f"{row.risk_reduction_pct} %")
            cols[2].markdown(
                f"<span style='background:#33415566;padding:4px 8px;"
                "border-radius:6px;font-size:11px'>Variant {row.variant}</span>",
                unsafe_allow_html=True,
            )

            with st.expander("📖 Rationale & Trade-offs", expanded=False):
                rationale_lines = [s.strip() for s in row.rationale.split(". ") if s]
                for r in rationale_lines:
                    st.markdown(f"• {r.strip().rstrip('.')}.")
                st.markdown(
                    f"**Cost:** {row.cost_pct_of_portfolio:.2f}% of capital  \n"
                    f"**Horizon:** {row.time_horizon_months} months  \n"
                    f"**Score:** {row.score:.2f}"
                )

                if st.button("✔️ Select this strategy", key=f"select_{i}"):
                    st.session_state.chosen_strategy = row.to_dict()
                    st.success(f"Selected **{row.name}**")

            st.markdown("</div>", unsafe_allow_html=True)

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
if "risk_cache"  not in st.session_state: st.session_state.risk_cache  = {}  # {ticker: [risks]}
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

from io import StringIO

uploaded_file = st.file_uploader("Upload your portfolio (CSV)", type=["csv"])
if uploaded_file:
    try:
        # Decode and read safely using the Python engine
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        df = pd.read_csv(StringIO(content), engine="python", on_bad_lines="warn")
    except Exception as e:
        st.error("❌ Error reading CSV. Please check for missing quotes, commas, or formatting issues.")
        st.exception(e)
        st.stop()

    required_cols = ["Ticker", "Amount ($)"]
    if not all(col in df.columns for col in required_cols):
        st.error("CSV must contain at least 'Ticker' and 'Amount ($)' columns.")
        st.stop()

    # Add Stop-Loss column if missing
    if "Stop-Loss ($)" not in df.columns:
        df["Stop-Loss ($)"] = None

    # Save cleaned DataFrame to session
    df["Ticker"] = df["Ticker"].astype(str).str.upper()
    st.session_state.alloc_df = df[["Ticker", "Amount ($)", "Stop-Loss ($)"]]
    st.session_state.portfolio = df["Ticker"].tolist()
    st.session_state.portfolio_alloc = dict(zip(df["Ticker"], df["Amount ($)"]))

else:
    # Fallback default portfolio
    if "alloc_df" not in st.session_state:
        st.session_state.alloc_df = pd.DataFrame({
            "Ticker": ["AAPL", "MSFT"],
            "Amount ($)": [10000, 10000],
            "Stop-Loss ($)": [None, None]
        })
    st.session_state.alloc_df = (
        st.session_state.alloc_df
        .query("Ticker in @st.session_state.portfolio")
        .sort_values("Amount ($)", ascending=False, ignore_index=True)
    )

# Clone the cleaned DataFrame for UI display
clean_df = st.session_state.alloc_df.copy()
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

st.dataframe(clean_df, use_container_width=True)

st.session_state.stop_loss_map = dict(
    zip(clean_df["Ticker"], clean_df["Stop-Loss ($)"])
)
st.session_state.alloc_df = clean_df
st.session_state.portfolio = clean_df["Ticker"].tolist()
st.session_state.portfolio_alloc = dict(
    zip(clean_df["Ticker"], clean_df["Amount ($)"])
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
    #risk_string = ", ".join(r for ticker in portfolio for r in st.session_state.risk_cache.get(ticker, [])[0]) or "None"
    risk_string = "; ".join(
        title for t in portfolio
        for title, _url in st.session_state.risk_cache.get(t, [])
    )
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

        # ─────────────────────────────────────────────────────────────────────────
    # NEW: ask the LLM for 3-4 hedging strategies in structured JSON
    #      (replaces the old “settings_prompt” + markdown-parsing flow)
    # ─────────────────────────────────────────────────────────────────────────
    import json, textwrap

    # Fallback sizing rules if you’re not already storing them in session
    hedge_budget_pct  = st.session_state.get("hedge_budget_pct",  2.0)  # %
    single_hedge_pct  = st.session_state.get("single_hedge_pct", 1.0)   # %

    SYSTEM_JSON = textwrap.dedent("""
    You are a senior equity strategist advising institutional investors.

    Return exactly one JSON object with a top-level key `strategies` (list of 3–4 items).
    Each item MUST include the following keys:

    - name: concise label (e.g. "SPY Put Spread", "VIX Call Hedge")
    - variant: string (e.g. "A", "B", "C") to distinguish alternatives
    - score: float from 0.0 to 1.0 (higher = more attractive)
    - risk_reduction_pct: integer (estimated VaR or drawdown reduction, 0–100%)
    - cost_pct_of_portfolio: float (capital required, % of notional)
    - time_horizon_months: integer (expected duration)
    - rationale: exactly 3 sentences:
    1. Describe the hedge and instrument mechanics clearly.
    2. Justify why this hedge is relevant based on market context (e.g. volatility skew, event risk, macro positioning).
    3. Note trade-offs: opportunity cost, convexity loss, or decay risk.

    Rank results by score descending.

    Do not use generalities like "market caution" or "Fed fading" unless tied to a specific instrument.
    Only return valid JSON. No markdown, no prose.
    """).strip()

    USER_JSON = textwrap.dedent(f"""
        Portfolio tickers: {', '.join(portfolio)}
        Allocations: {alloc_str}
        Total capital: ${total_capital:,.0f}
        Time horizon: {horizon} months
        Stop-loss triggers: {stop_loss_str or 'none'}
        Headline risk exposures: {risk_string or 'none'}
        Allowed hedge instruments: {', '.join(st.session_state.allowed_instruments)}

        Objective:
        Generate 3–4 differentiated hedge strategies that reduce downside risk exposure using liquid, cost-efficient instruments.
        Tailor hedges to real exposures (e.g., tech beta, small-cap skew, macro risk).
        Use realistic sizing and costs. Avoid duplication.
        Assume investor is familiar with options, futures, and ETF mechanics.

        Tone: concise, institutional, actionable.
        Return JSON only.
        """).strip()

    with st.spinner("⚙️  Generating multiple hedging strategies…"):
        raw_json = ask_openai(
            model=model,
            system_prompt=SYSTEM_JSON,
            user_prompt=USER_JSON,
        )

    # ───────────────────── parse & validate LLM result ─────────────────────
    try:
        data = json.loads(raw_json)
        df_strat = pd.DataFrame(data["strategies"])
    except (json.JSONDecodeError, KeyError) as err:
        st.error(f"❌ LLM returned invalid JSON: {err}")
        st.stop()

    # ─── Content quality check ───
    if df_strat["rationale"].str.len().mean() < 120:
        st.warning("⚠️ Strategy rationale looks too shallow. The LLM may have ignored the structure.")

    # Persist for downstream pages / reruns
    st.session_state.strategy_df = df_strat

    # ───────────────────── render basic “card list” ────────────────────────
    st.subheader("🛡️ Recommended Hedging Strategies")
    render_strategy_cards(df_strat)

    if st.session_state.chosen_strategy:
        st.info(f"**Chosen strategy:** {st.session_state.chosen_strategy['name']}")
    

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