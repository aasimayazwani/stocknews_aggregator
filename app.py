# app.py – Market-Movement Chatbot  (portfolio-aware + risk-scan edition)
from __future__ import annotations

import re, textwrap, requests
from typing import List
import requests
import os
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from config import DEFAULT_MODEL          # local module
from openai_client import ask_openai      # wrapper around OpenAI API
from stock_utils import get_stock_summary # your own helper
# ────────────────────────────────── THEME ─────────────────────────────────
st.set_page_config(page_title="Hedge Strategy Chatbot", layout="centered")

with st.sidebar.expander("📌 Investor Profile", expanded=False):
    st.radio("Experience", ["Beginner", "Intermediate", "Expert"], key="experience_level")
    st.radio("Detail level", ["Just the strategy", "Explain the reasoning", "Both"], key="explanation_pref")

with st.sidebar.expander("🧮 Investment Settings", expanded=True):
    st.selectbox("Focus stock", options=["AAPL", "MSFT", "TSLA"], key="focus_stock")
    st.slider("⏳ Time horizon (months)", 1, 24, 6, key="time_horizon")
    st.checkbox("🚫 Avoid suggesting same stocks in hedge", value=True, key="avoid_overlap")

with st.sidebar.expander("🎯 Hedge Instruments", expanded=False):
    st.multiselect(
        "Choose instruments to include:",
        options=["Put Options", "Collar Strategy", "Inverse ETFs", "Short Selling", "FX Options"],
        default=["Put Options", "Collar Strategy"],
        key="allowed_instruments"
    )

with st.sidebar.expander("⚙️ Risk & Budget Controls", expanded=False):
    st.slider("🎯 Beta match band", 0.5, 2.0, (1.15, 1.50), step=0.01, key="beta_band")
    st.slider("🔻 Stop-loss for shorts (%)", 1, 20, 10, key="stop_loss")
    st.slider("💰 Total hedge budget (% of capital)", 5, 25, 10, key="total_budget")
    st.slider("📉 Max per single hedge (% of capital)", 1, 10, 5, key="max_hedge")

with st.sidebar.expander("🧹 Session Tools", expanded=False):
    if st.button("🗑️ Clear Portfolio"):
        st.session_state.portfolio_alloc = {}
    if st.button("🧽 Clear Chat History"):
        st.session_state.chat_history = []

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
if "selected_risks" not in st.session_state: st.session_state.selected_risks = []

# ──────────────────────────── HELPERS ────────────────────────────────
def clean_md(md: str) -> str:
    md = re.sub(r"(\d)(?=[A-Za-z])", r"\1 ", md)
    return md.replace("*", "").replace("_", "")

def render_rationale(df: pd.DataFrame) -> None:
    """
    Show a nicely-formatted card for every hedge instrument
    in df.  Assumes columns: Ticker, Position, Amount ($),
    Rationale, Source   (anything missing is handled gracefully).
    """
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

        # add link if it looks like a URL
        if re.match(r'^https?://', src):
            card += f"<br><a href='{src}' target='_blank' style='color:#60a5fa;'>Source&nbsp;↗</a>"

        card += "</div>"
        st.markdown(card, unsafe_allow_html=True)


def fallback_ticker_lookup(name: str, model_name: str = "gpt-4.1-mini") -> str:
    prompt = f"What is the stock ticker symbol for the publicly traded company '{name}'?"
    raw = ask_openai(
        model=model_name,
        system_prompt="You are a financial assistant that returns only the correct stock ticker symbol.",
        user_prompt=prompt,
    )

    import re
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
        tickers = []
        for r in results:
            symbol = r.get("symbol", "")
            name = r.get("shortname") or r.get("longname") or ""
            if symbol and name:
                tickers.append(f"{symbol} – {name}")

        # Fallback: use GPT if nothing came back
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

import os
import requests
import streamlit as st

@st.cache_data(ttl=900, show_spinner=False)
def web_risk_scan(ticker: str):
    """
    Pulls analyst-related headlines for a given stock using NewsAPI.
    Deduplicates results and filters for risk-related language or known analysts.
    """
    api_key = st.secrets.get("NEWSAPI_KEY") or os.getenv("NEWSAPI_KEY")
    if not api_key:
        return [("⚠️ No NEWSAPI_KEY found. Please add it to .streamlit/secrets.toml", "#")]

    # Build query for ticker + analyst-related terms
    query = f'"{ticker}" AND (analyst OR downgrade OR rating OR earnings OR revise OR cut OR risk)'
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 15,
        "apiKey": api_key
    }

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
            continue  # Skip empty or duplicate titles

        # Check for risk-related or analyst headlines
        if any(keyword in title.lower() for keyword in [
            "downgrade", "risk", "cut", "concern", "slashed", "fall", "caution", "bearish", "revised"
        ]):
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
# 🛡️ Hedge instrument defaults by experience
experience_to_default = {
    "Beginner": ["Inverse ETFs", "Commodities"],
    "Intermediate": ["Put Options", "Inverse ETFs", "Commodities"],
    "Expert": ["Put Options", "Collar Strategy", "Inverse ETFs", "Short Selling", "Volatility Hedges", "Commodities", "FX Hedges"]
}

avoid_duplicate_hedges = st.session_state.get("avoid_overlap", True)
# Store in session state
st.session_state.avoid_dup_hedges = avoid_duplicate_hedges

# 🎯 basket computation moved below
others  = [t for t in st.session_state.portfolio if t != primary]
basket  = [primary] + others

# ───────────────────────────── PORTFOLIO UI ──────────────────────────
# ⬇️ NEW ticker search & autocomplete with live API results

# ─────────────────── 💰 POSITION-SIZE EDITOR ────────────────────
st.markdown("### 💰 Position sizes Editable")

# 1. Boot-strap a persistent table in session-state
if "alloc_df" not in st.session_state:
    tickers = st.session_state.portfolio
    st.session_state.alloc_df = pd.DataFrame({
    "Ticker": tickers,
    "Amount ($)": [10_000] * len(tickers),
    "Stop-Loss ($)": [None] * len(tickers)  # New column
})

# 2. Keep only valid rows
st.session_state.alloc_df = (
    st.session_state.alloc_df
      .query("Ticker in @st.session_state.portfolio")
      .sort_values("Amount ($)", ascending=False, ignore_index=True)
)

# 3–4. Clean and validate input from session state
clean_df = (
    st.session_state.alloc_df
      .dropna(subset=["Ticker"])
      .query("Ticker != ''")
      .drop_duplicates(subset=["Ticker"])
      .sort_values("Amount ($)", ascending=False, ignore_index=True)
)

# 5. Add real-time price and % change
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

# 6. Show a single data editor (with price columns read-only)
editor_df = st.data_editor(
    clean_df,
    disabled={"Price": True, "Δ 1d %": True},  # Stop-Loss is editable
    num_rows="dynamic",
    use_container_width=True,
    key="alloc_editor",
    hide_index=True,
)

# 7. Persist edits back to session state
st.session_state.stop_loss_map = dict(
    zip(editor_df["Ticker"], editor_df["Stop-Loss ($)"])
)
st.session_state.alloc_df = editor_df
st.session_state.portfolio = editor_df["Ticker"].tolist()
st.session_state.portfolio_alloc = dict(
    zip(editor_df["Ticker"], editor_df["Amount ($)"])
)

# 8. Create pie data (used conditionally later)
ticker_df = pd.DataFrame({
    "Ticker": list(st.session_state.portfolio_alloc.keys()),
    "Amount": list(st.session_state.portfolio_alloc.values())
}).sort_values("Amount", ascending=False)

ticker_df["Amount"] = ticker_df["Amount"].fillna(0)
ticker_df["Label"] = (
    ticker_df["Ticker"] + " ($" +
    ticker_df["Amount"].round(0).astype(int).astype(str) + ")"
)

# 10. Save final list
portfolio = st.session_state.portfolio

# ────────────────────── AUTOMATED RISK SCAN SECTION ───────────────────
st.markdown("### 🔍  Key headline risks")
if primary not in st.session_state.risk_cache:
    with st.spinner("Scanning web…"):
        st.session_state.risk_cache[primary] = web_risk_scan(primary)



risk_tuples = st.session_state.risk_cache[primary]      # output of web_risk_scan()
risk_titles = [t[0] for t in risk_tuples]               # e.g. "FTC opens probe…"
risk_links  = {title: url for title, url in risk_tuples}

# 🚫 Skip risk rendering if no real headlines found
if len(risk_titles) == 1 and "#".startswith(risk_links.get(risk_titles[0], "#")):
    st.info(risk_titles[0])
else:
    # 🟢 Only run this block if real risks are found
    selected_risks = []
    risk_links = {r: f"https://www.google.com/search?q={primary}+{r.replace(' ', '+')}" for r in risk_titles}
    st.markdown("<div class='risk-grid'>", unsafe_allow_html=True)
    
    for i, risk in enumerate(risk_titles):
        if "no fresh negative headlines" in risk.lower():
            continue  # Don't render dummy message as a checkbox
        key = f"risk_{i}"
        if key not in st.session_state:
            st.session_state[key] = True
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
        if st.session_state[key]:
            selected_risks.append(risk)

    st.markdown("</div>", unsafe_allow_html=True)
    st.session_state.selected_risks = selected_risks
    st.session_state.risk_ignore = [r for r in risk_titles if r not in selected_risks]


st.markdown("Un-check any headline you **do not** want the LLM to consider:")

# --------------------------------------------
# 🧠 2-Column Responsive Risk Rendering Section
# --------------------------------------------
selected_risks = []

# Begin the grid container
st.markdown("<div class='risk-grid'>", unsafe_allow_html=True)

# Render each risk in a styled card with checkbox + real source link
for i, risk in enumerate(risk_titles):
    key = f"risk_{i}"
    
    # Set default checkbox state to True (checked) on first render
    if key not in st.session_state:
        st.session_state[key] = True

    # Read current state and reflect in HTML attribute
    checked_attr = "checked" if st.session_state[key] else ""

    # Safely look up associated URL (fallback = "#")
    link = risk_links.get(risk, "#")

    # Render a styled checkbox card with clickable ℹ️ icon
    html = f"""
    <div class='risk-card'>
      <label for="{key}">
        <input type="checkbox" id="{key}" name="{key}" {checked_attr}>
        <span>{risk}</span>
        <a href="{link}" target="_blank">ℹ️</a>
      </label>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

    # Track selected risks in session state
    if st.session_state[key]:
        selected_risks.append(risk)

st.session_state.selected_risks = selected_risks
# End the grid
st.markdown("</div>", unsafe_allow_html=True)

# Update the exclusion list in session state
st.session_state.risk_ignore = [r for r in risk_titles if r not in selected_risks]


# 🔄  Store & show sticky pill
st.session_state.experience_level  = experience_level
st.session_state.explanation_pref  = explanation_pref

experience_to_default = {
    "Beginner": ["Inverse ETFs", "Commodities"],
    "Intermediate": ["Put Options", "Inverse ETFs", "Commodities"],
    "Expert": ["Put Options", "Collar Strategy", "Inverse ETFs", "Short Selling", "Volatility Hedges", "Commodities", "FX Hedges"]
}

default_instruments = experience_to_default.get(st.session_state.experience_level, [])

# Sidebar: Hedge instruments based on experience

if "strategy_history" not in st.session_state:
    st.session_state.strategy_history = []

if st.button("🗑️ Clear Strategy History"):
    st.session_state.strategy_history = []
    st.rerun()




# ─────────────────────────── STRATEGY DESIGNER ───────────────────────
st.markdown("### 📝  Strategy Designer")
#sector_guess = yf.Ticker(primary).info.get("sector", "")
#sector_in    = st.text_input("Sector", sector_guess)
#goal         = st.selectbox("Positioning goal", ["Long", "Short", "Hedged", "Neutral"])
#avoid_sym    = st.text_input("Hedge / avoid ticker", primary)
#capital      = st.number_input("Capital (USD)", 1000, 1_000_000, 10_000, 1000)
#horizon      = st.slider("Time horizon (months)", 1, 24, 6)

# Strategy generation & rendering
if st.button("Suggest strategy", type="primary"):
    # 1. Collect context values
    ignored = "; ".join(st.session_state.risk_ignore) or "None"
    total_capital = sum(st.session_state.portfolio_alloc.values())
    risk_string = ", ".join(risk_titles) or "None"
    alloc_str = "; ".join(f"{k}: ${v:,.0f}" for k, v in st.session_state.portfolio_alloc.items()) or "None"

    # 2. Dynamic tone-/length guidance
    st.session_state.experience_level = experience_level
    st.session_state.explanation_pref = explanation_pref
    exp_pref = st.session_state.explanation_pref

    experience_note = {
        "Beginner":     "Use plain language and define jargon the first time you use it.",
        "Intermediate": "Assume working knowledge of finance; keep explanations concise.",
        "Expert":       "Write in professional sell-side style; no hand-holding.",
    }[st.session_state.experience_level]

    if exp_pref == "Just the strategy":
        rationale_rule = "Each *Rationale* must be **≤ 25 words (one sentence)**."
    elif exp_pref == "Explain the reasoning":
        rationale_rule = ("Each *Rationale* must be **2 sentences totalling ≈ 30-50 words** "
                          "(logic + risk linkage).")
    else:  # "Both"
        rationale_rule = ("Each *Rationale* must be **3 sentences totalling ≈ 60-90 words** – "
                          "1️⃣ logic, 2️⃣ quantitative context, 3️⃣ trade-offs.")

    stop_loss_str = "; ".join(
        f"{ticker}: ${float(sl):.2f}" for ticker, sl in st.session_state.stop_loss_map.items() if pd.notnull(sl)
    ) or "None"

    hedge_budget_pct = st.session_state.hedge_budget_pct
    single_hedge_pct = st.session_state.single_hedge_pct
    max_hedge_notional = total_capital * hedge_budget_pct / 100

    avoid_note = ""
    if st.session_state.avoid_dup_hedges:
        avoid_note = (
            "- ❌ **Do NOT suggest hedge instruments already in the user’s portfolio** "
            f"({', '.join(st.session_state.portfolio)}).\n"
            "- ✅ Prefer diversifiers (sector ETFs, index futures, inverse ETFs, FX, commodities).\n"
        )

    # Build final prompt
    prompt = textwrap.dedent(f"""
        Act as a **tactical hedging strategist**.  
        Goal: *preserve capital* while keeping portfolio beta inside **{beta_rng[0]:.2f}–{beta_rng[1]:.2f}**.

        {avoid_note}
        **Allowed hedge types**: {', '.join(st.session_state.allowed_instruments)}
        Only suggest instruments from the list above. Do **NOT** suggest anything not listed.

        ### Step-by-Step Reasoning
        1. **Identify Hedging Targets**  
        • Flag holdings with  
            – Stop-loss ≥ 5 % above market price, **or**  
            – High sensitivity to headline risks: {risk_string}.  
        • Ignore: {ignored}

        2. **Select Instruments**  
        • Primary – Put options (strike ≤ stop-loss – 2 %; expiry {horizon} ± 0.5 mo).  
        • Secondary – Shorts / inverse ETFs / futures **only** if stop-loss buffer ≥ {stop_loss} %.

        3. **Size Positions**  
        • **Total hedge budget ≤ {hedge_budget_pct} %** of capital (${total_capital:,.0f}).  
        • **Any single hedge ≤ {single_hedge_pct} %** of capital.  
        • Rebalance to maintain target beta.

        4. **Cost Optimisation**  
        • Aim for option premium ≤ 3 % of notional per hedge.

        ---
        **Context snapshot**  
        • Basket: {', '.join(basket)}  
        • Allocation: {alloc_str}  
        • User stop-losses: {stop_loss_str}  
        • Horizon: {horizon} mo  
        • Total capital: ${total_capital:,.0f}  
        • Portfolio stop-loss buffer (shorts): {stop_loss} %

        ### Investor profile  
        Experience: {st.session_state.experience_level} • Detail: {st.session_state.explanation_pref} → {experience_note}

        ---
        ### OUTPUT SPEC *(Markdown only — no tables, no code fences, no HTML)*

        **Hedge list** — one numbered bullet per hedge, *exactly* like this template  
        ```
        1. **AAPL** — Put Option. Buy 3 × Aug $175 puts … (≤ {rationale_rule}) [1]  
        2. **MSFT** — Short via PSQ ETF … [2]  
        ```

        **Sizing table** — immediately *after* the bullets  
        ```
        | Hedge | Qty / Contracts | $ Notional | % of Capital |
        |-------|-----------------|-----------:|-------------:|
        | AAPL Aug 175 Puts | 3 contracts | $3 000 | 3 % |
        | PSQ ETF          | 500 sh       | $5 000 | 5 % |
        | …                | …            | …      | … |
        | **Total**        |              | **≤ ${max_hedge_notional:,.0f}** | **≤ {hedge_budget_pct}%** |
        ```
        • Ensure each row ≤ {single_hedge_pct}% and total row ≤ {hedge_budget_pct}%.  
        • If you recommend futures, show contract size equal to the notional you quote.

        **Rules**  
        • Each bullet ends with a reference marker like `[1]`.  
        • {rationale_rule}  
        • If an option, include strike / expiry / premium / # contracts in the bullet.  
        • Do **not** suggest tickers already in the user’s portfolio if diversification is required.  
        • Limit to **5 hedges maximum**.

        **Reference list** (immediately after the sizing table, one per line)  
        ```
        [1] https://valid.source/for/aapl  
        [2] https://another.source/example  
        ```

        **After the references**, add  
        1. `### Summary` — ≤ 300 characters.  
        2. `### Residual Risks` — numbered list, each risk ≤ 25 words **and** ending with its own URL.

        ❗ Final answer: plain Markdown only.
    """).strip()

    # 2.  Call OpenAI -----------------------------------------------------------
    with st.spinner("Calling ChatGPT…"):
        raw_md = ask_openai(model, "You are a precise, citation-rich strategist.", prompt)

    plan_md = raw_md

    # Extract footnotes once: [1] https://...
    footnotes = dict(re.findall(r"\[(\d+)\]\s+(https?://[^\s]+)", plan_md))

    # Replace [1], [2], etc. with markdown-embedded superscript links
    superscripts_map = "⁰¹²³⁴⁵⁶⁷⁸⁹"
    for ref_id, url in footnotes.items():
        superscript = "".join(superscripts_map[int(d)] for d in ref_id)
        plan_md = plan_md.replace(f"[{ref_id}]", f"[🔗{superscript}]({url})")

    # Remove raw footnote lines like: [1] https://...
    plan_md = re.sub(r"^\[\d+\]\s+https?://[^\s]+", "", plan_md, flags=re.MULTILINE)

    # Define hedge_lines for downstream use
    lines = plan_md.splitlines()
    hedge_lines = [line for line in lines if re.match(r"^\d+\.\s+\*\*.+\*\*", line)]

    # Display strategy
    st.subheader("📌 Suggested strategy")


    records = []
    for line in hedge_lines:
        #match = re.match(r"^(\d+)\.\s+\*\*(.+?)\*\*\s+—\s+(.*?)\s+\[(\d+)\]", line)
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

    # 📌 User table (from editor)
    user_df = editor_df.copy()
    user_df["Position"] = "Long"
    user_df["Source"] = "User portfolio"
    user_df["% of Portfolio"] = (user_df["Amount ($)"] / user_df["Amount ($)"].sum() * 100).round(2)
    user_df["Rationale"] = "—"
    user_df["Ticker"] = user_df["Ticker"].astype(str)

    # 🧮 Optional: combine user + hedge tables
    df["% of Portfolio"] = 0
    df["Price"] = "_n/a_"
    df["Δ 1d %"] = "_n/a_"
    user_df["Price"] = user_df["Price"].round(2)
    user_df["Δ 1d %"] = user_df["Δ 1d %"].round(2)

    final_cols = ["Ticker", "Position", "Amount ($)", "% of Portfolio", "Price", "Δ 1d %", "Source", "Rationale"]
    user_df = user_df[final_cols]
    missing_cols = [col for col in final_cols if col not in df.columns]
    for col in missing_cols:
        df[col] = ""  # fill with empty strings

    # Now safe to reorder
    df = df[final_cols]
    combined_df = pd.concat([user_df, df], ignore_index=True)

    # ✅ Store in session state
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

    # 📊 Post-hedge allocation (optional chart)
    st.dataframe(combined_df.drop(columns=["Rationale"]), use_container_width=True)

    # ✅ Markdown rationale display (not the table)
    st.markdown("### 📌 Hedge Strategy Rationale")
    st.markdown(plan_md)

# ─────────────────────────────── STRATEGY HISTORY ───────────────────────────────
st.markdown("### 🕘 Previous Strategies")

if not st.session_state.strategy_history:
    st.info("No previous strategies yet.")
else:
    for idx, run in reversed(list(enumerate(st.session_state.strategy_history))):
        with st.expander(f"Run {idx+1} — {run['timestamp']} | Horizon: {run['horizon']} mo"):
            st.markdown(
                f"**Capital**: ${run['capital']:,.0f} • "
                f"**Beta Band**: {run['beta_band'][0]}–{run['beta_band'][1]}"
            )
            st.dataframe(run["strategy_df"], use_container_width=True)
            st.markdown(run["rationale_md"])

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
