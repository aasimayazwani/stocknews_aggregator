from __future__ import annotations
import re, textwrap, requests
import json, textwrap
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
        label="Investor experience", 
        options=["Beginner", "Intermediate", "Expert"],
        index=["Beginner", "Intermediate", "Expert"].index(st.session_state.get("experience_level", "Expert")),
        format_func=lambda x: f"Experience: {x}",
        key="experience_level"
    )

    # Detail level
    detail_level = st.selectbox(
        label="Explanation detail preference",
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

# ──────────────────────────── HELPERS ────────────────────────────────
def render_strategy_cards(df: pd.DataFrame) -> None:
    """
    Render strategy cards.
    Title now shows a 4–5-word justification headline instead of the raw name.
    """
    if df.empty:
        st.info("No strategies available.")
        return

    for i, row in df.iterrows():
        # ── create 4-5-word headline from first sentence of rationale ──────
        raw_rationale = row.rationale
        thesis_text = raw_rationale.get("thesis") if isinstance(raw_rationale, dict) else str(raw_rationale)
        first_sentence = thesis_text.split(".")[0].strip()
        headline_words = first_sentence.split()[:5]
        headline = " ".join(headline_words) + "…"
        # ── highlight if selected ─────────────────────────────────────────
        chosen   = st.session_state.get("chosen_strategy") or {}
        selected = chosen.get("name") == row.name
        box_color = "#10b981" if selected else "#334155"

        with st.container():
            st.markdown(
                f"""
                <div style="
                    border: 1px solid {box_color};
                    border-radius: 10px;
                    padding: 16px;
                    margin-bottom: 16px;
                    background-color: #1e1f24;
                ">
                <div style="display:flex; justify-content: space-between; align-items:center;">
                    <div style="font-size: 18px; font-weight: 600;">{headline}</div>
                    <div style="font-size: 13px; background-color: #334155;
                        color: #f8fafc; padding: 4px 10px; border-radius: 6px;">
                        Variant {row.variant}
                    </div>
                </div>

                <div style="margin-top: 8px;">
                    <b>Risk Reduction:</b> {row.risk_reduction_pct}% &nbsp;&nbsp;
                    <b>Cost:</b> {row.get("aggregate_cost_pct", 0):.1f}% of capital &nbsp;&nbsp;
                    <b>Horizon:</b> {row.get("horizon_months", "—")} months
                </div>


                <details style="margin-top: 12px; color: #e2e8f0;">
                    <summary style="cursor: pointer;">📖 View Rationale & Trade-offs</summary>
                    <div style="margin-top: 8px; line-height: 1.6;">
                        {(
                            f"• {raw_rationale.get('thesis', '').rstrip('.')}.<br>• {raw_rationale.get('tradeoff', '').rstrip('.')}"
                            if isinstance(raw_rationale, dict)
                            else "<br>".join(f"• {s.strip()}." for s in str(raw_rationale).split(".") if s.strip())
                        )}
                    </div>
                    <form method="post">
                        <button type="submit"
                            style="
                                margin-top: 12px;
                                padding: 6px 12px;
                                background-color: #10b981;
                                color: white;
                                border: none;
                                border-radius: 6px;
                                cursor: pointer;
                            "
                            name="select_strategy_{i}"
                        >✔️ Select this strategy</button>
                    </form>
                </details>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # update selection flag if user clicked the button
            if st.session_state.get(f"select_strategy_{i}"):
                st.session_state.chosen_strategy = row.to_dict()

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
    df = yf.download(tickers, period=period, progress=False, auto_adjust=True)["Close"]
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
                is_selected = st.checkbox(
                    label=f"Select: {risk}",   # ← FIXED
                    key=key,
                    value=default,
                    label_visibility="collapsed"  # Optional if you want to hide visually
                )
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

    def _length(x):
        return len(x.get("thesis", "") + x.get("tradeoff", "")) if isinstance(x, dict) else len(str(x))

    def avg_stop_loss_pct(df: pd.DataFrame) -> str:
        pct_list = []
        for _, row in df.iterrows():
            sl = row["Stop-Loss ($)"]
            px = row["Price"]
            if pd.notnull(sl) and px and px > 0:
                pct_list.append(abs(px - sl) / px * 100)
        return f"{round(sum(pct_list) / len(pct_list), 1)}" if pct_list else "n/a"

    # Fallback sizing rules if you’re not already storing them in session
    hedge_budget_pct  = st.session_state.get("hedge_budget_pct",  2.0)  # %
    single_hedge_pct  = st.session_state.get("single_hedge_pct", 1.0)   # %

    SYSTEM_JSON = textwrap.dedent("""
        You are a senior equity-derivatives strategist.

        Return ONE valid JSON object:

        strategies: [            # 3-4 ideas, ranked by score
        {
            name:  string,
            variant: string,           # A / B / C
            score:  float,             # 0-1
            risk_reduction_pct: int,   # VaR or max-drawdown Δ
            horizon_months: int,       # overall hedge horizon
            legs: [                    # 1-3 hedge legs
            {
                instrument: string,        # e.g. "SPY Sep 430P", "ESU4 future"
                position:  string,         # "long", "short", "ratio 2:1"
                notional_pct: float,       # % of portfolio notionally hedged
                cost_pct_capital: float,   # premium or margin as % of capital
                expiry:  string            # "2025-09-20", or "3m"
            }
            ],
            aggregate_cost_pct: float,     # sum of leg costs
            rationale: {                   # keep it tight
            thesis:   string,            # ≤ 25 words
            tradeoff: string             # ≤ 20 words
            }
        }
        ]

        Return JSON only.
        """).strip()

    USER_JSON = textwrap.dedent(f"""
        Portfolio tickers: {', '.join(portfolio)}
        Allocations: {alloc_str}
        Total capital: ${total_capital:,.0f}
        Time horizon: {horizon} months
        Stop-loss triggers: {stop_loss_str or 'none'}
        Headline risk exposures: {risk_string or 'none'}
        Allowed hedge instruments: {', '.join(st.session_state.allowed_instruments)}

        Holdings: {alloc_str}
        Capital: ${total_capital:,.0f}
        Average stop-loss distance: {avg_stop_loss_pct}%
        Key risks: {risk_string or "none"}
        Allowed instruments: {', '.join(st.session_state.allowed_instruments)}

        Requirements:
        • 1-3 hedge legs per strategy
        • Total premium ≤ {hedge_budget_pct}% of capital
        • Legs may hedge at index, sector, or single-name level
        • Show expiry explicitly (e.g. Sep-24).

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
        if not raw_json.strip().startswith("{"):
            st.error("❌ LLM did not return valid JSON.")
            st.code(raw_json.strip() or "[Empty response]", language="text")
            st.stop()
    # ───────────────────── parse & validate LLM result ─────────────────────
    try:
        data = json.loads(raw_json)
        # ---- split top-level metrics and legs -------------------------------
        df_strat = pd.DataFrame(                        # ← keep this name so the rest
            [{k: v for k, v in s.items() if k != "legs"}  # of your code (cards etc.)
            for s in data["strategies"]]                # doesn’t need edits
        )

        # store legs per strategy index
        st.session_state.strategy_legs = {
            idx: s["legs"] for idx, s in enumerate(data["strategies"])
        }
    except (json.JSONDecodeError, KeyError) as err:
        st.error(f"❌ LLM returned invalid JSON: {err}")
        st.stop()

    # ─── Content quality check ───
    if df_strat["rationale"].apply(_length).mean() < 120:
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