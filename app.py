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

@st.cache_data(ttl=900, show_spinner=False)
def web_risk_scan(ticker: str) -> list[tuple[str, str]]:
    """
    Enhanced news-based risk scanner using NewsAPI,
    looking for analyst-related and market-moving content.
    Returns: List of (title, url) tuples.
    """
    import os
    from datetime import datetime, timedelta

    api_key = st.secrets.get("NEWSAPI_KEY") or os.getenv("NEWSAPI_KEY")
    if not api_key:
        return [("NEWSAPI key missing", "#")]

    query = f"{ticker} analyst OR earnings OR downgrade OR forecast OR target"
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={query}&"
        "language=en&sortBy=publishedAt&pageSize=20&"
        f"from={(datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')}&"
        f"apiKey={api_key}"
    )

    try:
        resp = requests.get(url, timeout=10)
        articles = resp.json().get("articles", [])
    except Exception as e:
        return [(f"News API error: {e}", "#")]

    known_analysts = {"Dan Ives", "Mark Mahaney", "Katy Huberty", "Gene Munster"}

    risks = []
    for article in articles:
        title = article.get("title", "")
        url = article.get("url", "#")
        if any(keyword in title.lower() for keyword in ["downgrade", "risk", "cut", "concern", "slashed", "fall", "caution", "bearish", "revised"]):
            risks.append((title, url))
        elif any(analyst in title for analyst in known_analysts):
            risks.append((f"📊 Analyst: {title}", url))

        if len(risks) >= 5:
            break

    return risks or [(f"No relevant analyst headlines found for {ticker}.", "#")]


# ─────────────────────────────── STATE ────────────────────────────────
if "history"     not in st.session_state: st.session_state.history     = []
if "portfolio"   not in st.session_state: st.session_state.portfolio   = ["AAPL", "MSFT"]
if "outlook_md"  not in st.session_state: st.session_state.outlook_md  = None
if "risk_cache"  not in st.session_state: st.session_state.risk_cache  = {}
if "risk_ignore" not in st.session_state: st.session_state.risk_ignore = []

# ────────────────────────── SIDEBAR – SETTINGS ───────────────────────
with st.sidebar.expander("⚙️  Settings"):
    model = st.selectbox("OpenAI Model", [DEFAULT_MODEL, "gpt-4.1-mini", "gpt-4o-mini"], 0)
    if st.button("🧹  Clear chat history"):  st.session_state.history = []
    if st.button("🗑️  Clear portfolio"):    st.session_state.portfolio = []

# fix duplicate ID bug by giving a key to each sidebar widget
with st.sidebar.expander("🕒 Investment settings", expanded=True):
    primary = st.selectbox("🎯 Focus stock", st.session_state.portfolio, 0, key="focus_stock")
    horizon = st.slider("⏳ Time horizon (months)", 1, 24, 6, key="horizon_slider")

show_charts = st.sidebar.checkbox("📈  Show compar-chart", value=False, key="show_chart_toggle")

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

# 9. Optional pie chart toggle
with st.sidebar:
    if st.checkbox("📊 Show Portfolio Pie Chart", value=False, key="sidebar_portfolio_pie"):
        st.markdown("#### 🥧 Portfolio Allocation")
        st.plotly_chart(
            px.pie(
                ticker_df,
                names="Label",
                values="Amount",
                hole=0.3
            ).update_traces(textinfo="label+percent"),
            use_container_width=True
        )

# 10. Save final list
portfolio = st.session_state.portfolio

# ────────────────────── headline-risk retrieval (cached) ─────────────────────
if primary not in st.session_state.risk_cache:
    with st.spinner("Scanning news with ChatGPT…"):
        st.session_state.risk_cache[primary] = web_risk_scan(primary)

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


# investor profile
# ▶️  right after st.sidebar.expander("⚙️  Settings"):
with st.sidebar.expander("🧑‍💼  Investor profile", expanded=True):
    experience_level   = st.radio("Experience",   ["Beginner", "Intermediate", "Expert"])
    explanation_pref   = st.radio("Detail level", ["Just the strategy", "Explain the reasoning", "Both"])

# 🔄  Store & show sticky pill
st.session_state.experience_level  = experience_level
st.session_state.explanation_pref  = explanation_pref

if "strategy_history" not in st.session_state:
    st.session_state.strategy_history = []

if st.button("🗑️ Clear Strategy History"):
    st.session_state.strategy_history = []
    st.rerun()

st.sidebar.markdown(
    f"<div style='margin-top:6px;padding:4px 8px;border-radius:12px;"
    f"background:#334155;color:#f8fafc;display:inline-block;font-size:13px;'>"
    f"{experience_level} • {explanation_pref}</div>",
    unsafe_allow_html=True,
)


# ─────────────────────────── STRATEGY DESIGNER ───────────────────────
st.markdown("### 📝  Strategy Designer")
#sector_guess = yf.Ticker(primary).info.get("sector", "")
#sector_in    = st.text_input("Sector", sector_guess)
#goal         = st.selectbox("Positioning goal", ["Long", "Short", "Hedged", "Neutral"])
#avoid_sym    = st.text_input("Hedge / avoid ticker", primary)
#capital      = st.number_input("Capital (USD)", 1000, 1_000_000, 10_000, 1000)
#horizon      = st.slider("Time horizon (months)", 1, 24, 6)

with st.expander("⚖️  Risk controls"):
    beta_rng  = st.slider("Beta match band", 0.5, 1.5, (0.8, 1.2), 0.05)
    stop_loss = st.slider("Stop-loss for shorts (%)", 1, 20, 10)

# ─────────────────────── Strategy generation & rendering ───────────────────────
if st.button("Suggest strategy", type="primary"):
    # ─── 1. Collect context values ───────────────────────────
    ignored        = "; ".join(st.session_state.risk_ignore) or "None"
    total_capital  = sum(st.session_state.portfolio_alloc.values())
    risk_string    = ", ".join(risk_titles) or "None"
    alloc_str      = "; ".join(f"{k}: ${v:,.0f}" 
                               for k,v in st.session_state.portfolio_alloc.items()) or "None"

    # ─── 2.  🔹 NEW: dynamic tone-/length guidance  ───────────
    experience_note = {
        "Beginner":     "Use plain language and define jargon the first time you use it.",
        "Intermediate": "Assume working knowledge of finance; keep explanations concise.",
        "Expert":       "Write in professional sell-side style; no hand-holding.",
    }[st.session_state.experience_level]

    exp_pref = st.session_state.explanation_pref
    if exp_pref == "Just the strategy":
        rationale_rule = "Each *Rationale* must be **≤ 25 words (one sentence)**."
    elif exp_pref == "Explain the reasoning":
        rationale_rule = ("Each *Rationale* must be **2 sentences totalling ≈ 30-50 words** "
                          "(logic + risk linkage).")
    else:                   # "Both"
        rationale_rule = ("Each *Rationale* must be **3 sentences totalling ≈ 60-90 words** – "
                          "1️⃣ logic, 2️⃣ quantitative context, 3️⃣ trade-offs.")
    stop_loss_str = "; ".join(
        f"{ticker}: ${float(sl):.2f}" for ticker, sl in st.session_state.stop_loss_map.items() if pd.notnull(sl)
    ) or "None"
    prompt = textwrap.dedent(f"""
        Act as a **tactical hedging strategist**.

        • **Basket**: {', '.join(basket)}
        • **Current allocation**: {alloc_str}
        • **Total capital**: ${total_capital:,.0f}
        • **User-defined stop-loss levels**: {stop_loss_str}
        • **Horizon**: {horizon} months
        • **Beta band**: {beta_rng[0]:.2f}–{beta_rng[1]:.2f}
        • **Portfolio-level stop-loss buffer** (shorts only): {stop_loss} %
        • **Detected headline risks for {primary}**: {risk_string or 'None'}
        • **Ignore**: {ignored or 'None'}

        ### Investor profile
        Experience: {st.session_state.experience_level}   •  Detail level: {exp_pref}
        → {experience_note}

        ---
        ### OUTPUT SPEC — *Markdown only*

        | Ticker | Position | Amount ($) | Rationale | Source |
        |--------|----------|------------|-----------|--------|

        **Rationale requirements**  
        {rationale_rule}

        **If an entry uses options, you must include all five bullet-points below**  
        1. **Option type** (Put/Call)  
        2. **Strike price** (⚠️ *at or just below the user’s stop-loss level*)  
        3. **Expiration date** (e.g. *16 Aug 2025, 30 DTE*)  
        4. **Approx. premium per contract** (USD)  
        5. **# Contracts** (justify sizing vs. underlying notional)  

        End every rationale with a citation tag like **[1]** that matches the URL in *Source*.

        *Source* column = exactly one live, clickable URL per row—no extra text.

        After the table add:  
        1. `### Summary` — ≤ 300 chars.  
        2. `### Residual Risks` — numbered list, ≤ 25 words each, each ending with its own URL.

        ---
        #### 📝 FORMAT EXAMPLE – FOR REFERENCE ONLY (DO NOT copy verbatim)

        | Ticker | Position | Amount ($) | Rationale | Source |
        |--------|----------|------------|-----------|--------|
        | AAPL | **Put Option** | 1,000 | Buy 3 × Aug $175 puts (≈ $2.30 / c) to cap downside below $172 stop-loss; 1-month horizon aligns with earnings gap risk. **[1]** | https://finance.yahoo.com/quote/AAPL/options |
        | MSFT | Long | 9,000 | Maintain core stake; minor trim funds puts while preserving upside vs. AI catalysts. **[2]** | https://www.cnbc.com/2025/07/01/microsoft-ai-outlook.html |

        ### Summary  
        Suggests tight put hedges at strikes just under stop-losses; retains core upside while capping downside to –5 %.

        ### Residual Risks  
        1. Vol crush reduces hedge efficacy if implied vols retrace. https://www.cboe.com  
        2. Macro shock may breach put strikes before adjustment window. https://www.federalreserve.gov
        ---

        Use the above example **only** to mirror structure, level of detail, and option specificity.  
        ❗ Absolutely **do not** wrap your final answer in code fences or quotes.
        """).strip()



    # 2.  Call OpenAI -----------------------------------------------------------
    with st.spinner("Calling ChatGPT…"):
        raw_md = ask_openai(model, "You are a precise, citation-rich strategist.", prompt)

    # 3.  Clean & show ----------------------------------------------------------
    plan_md = clean_md(raw_md)
    
        # Extract plan sections
    plan_md_main = re.sub(r"### Residual Risks.*", "", plan_md, flags=re.I | re.S)
    st.subheader("📌 Suggested strategy")

    # Instead of showing plan_md_main directly, parse & integrate the table
    md_lines = plan_md_main.splitlines()
    table_lines = [line for line in md_lines if '|' in line and not line.startswith('###')]

    if len(table_lines) >= 3:
        try:
            import io

            # STEP 1: Parse markdown table
            table_str = '\n'.join(table_lines)
            df = pd.read_csv(io.StringIO(table_str), sep='|')
            df.columns = [c.strip() for c in df.columns]
            df = df.dropna(subset=['Ticker', 'Amount ($)'])
            df = df.dropna(how="all", axis=1)                       # remove unnamed first col if present
            df = df[~df["Ticker"].str.contains(r"^-+|Ticker", na=False)]

            # Clean amounts
            df["Amount ($)"] = (
                df["Amount ($)"]
                .astype(str)
                .str.replace("$", "")
                .str.replace(",", "")
                .str.extract(r"(\d+\.?\d*)")[0]
                .astype(float)
            )

            # Add hedge table extras
            df["Price"] = "_n/a_"
            df["Δ 1d %"] = "_n/a_"
            df["Source"] = "Suggested hedge"
            total_amount = df["Amount ($)"].sum()
            df["% of Portfolio"] = (df["Amount ($)"] / total_amount * 100).round(2)

            # Now process user table
            user_df = editor_df.copy()
            user_df["Position"] = "Long"
            user_df["Source"] = "User portfolio"
            user_total = user_df["Amount ($)"].sum()
            user_df["% of Portfolio"] = (user_df["Amount ($)"] / user_total * 100).round(2)
            user_df["Rationale"] = "—"
            user_df["Ticker"] = user_df["Ticker"].astype(str)

                    # 🔄 Align columns for merging
            # Define base columns
            user_cols  = ["Ticker", "Position", "Amount ($)", "% of Portfolio", "Price", "Δ 1d %", "Source"]
            hedge_cols = user_cols + ["Rationale"]

            # 1️⃣ Keep hedge_df as is (includes rationale)
            df = df[hedge_cols]

            # 2️⃣ Prepare user_df (exclude rationale from display)
            user_df = user_df[user_cols].copy()

            # 3️⃣ Add empty rationale column for alignment only
            user_df["Rationale"] = ""

            # 4️⃣ Final unified ordering
            final_cols = hedge_cols
            user_df = user_df[final_cols]
            df = df[final_cols]

            # ✅ Store in session state
            st.session_state.user_df = user_df
            st.session_state.strategy_history.append({
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "horizon": horizon,
                "beta_band": beta_rng,
                "capital": total_capital,
                "strategy_df": df,
                "rationale_md": plan_md,  # full rationale markdown
            })

            # 📌 Combine for final rendering (if needed)
            combined_df = pd.concat([user_df, df], ignore_index=True)
            display_df = combined_df.drop(columns=["Rationale"])
            st.dataframe(display_df, use_container_width=True)
            #st.session_state.combined_df = combined_df

            # ✅ Guard in case data becomes stale or corrupted
            if combined_df.empty:
                st.warning("Combined hedge strategy is empty. Please re-generate.")
            else:
                pass
                #st.dataframe(combined_df, use_container_width=True)


            with st.sidebar:
                if st.checkbox("📊 Show Post-Hedge Pie Chart", value=False, key="sidebar_post_hedge_pie"):
                    st.markdown("#### 🧮 Post-Hedge Allocation")
                    pie_df = combined_df.copy()
                    pie_df["Label"] = pie_df["Ticker"] + " (" + pie_df["Position"] + ")"
                    pie_df["Amount"] = pie_df["Amount ($)"]
                    st.plotly_chart(
                        px.pie(
                            pie_df,
                            names="Label",
                            values="Amount",
                            hole=0.3
                        ).update_traces(textinfo="label+percent"),
                        use_container_width=True
                    )

        except Exception as e:
            st.warning(f"Could not parse or merge hedge table: {e}")

        # 🔍 📌 Hedge Strategy Rationale (dynamic, styled, and link-aware)
    st.markdown("### 📌 Hedge Strategy Rationale")

    hedge_df = st.session_state.get("strategy_df")   # <— safe fetch
    if hedge_df is None or hedge_df.empty:
        st.info("No hedge rationale to show.")
    else:
        render_rationale(hedge_df)  

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
