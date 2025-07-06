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
        "Amount ($)": [10_000] * len(tickers)
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
    disabled={"Price": True, "Δ 1d %": True},
    num_rows="dynamic",
    use_container_width=True,
    key="alloc_editor",  # single key now
    hide_index=True,
)

# 7. Persist edits back to session state
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
        st.session_state.risk_cache[primary] = web_risk_scan(primary, model)

# ────────────────────── AUTOMATED RISK SCAN SECTION ───────────────────
st.markdown("### 🔍  Key headline risks")

if primary not in st.session_state.risk_cache:
    with st.spinner("Scanning web…"):
        st.session_state.risk_cache[primary] = web_risk_scan(primary)

risk_list = st.session_state.risk_cache[primary]
selected_risks = st.session_state.get("selected_risks", risk_list)
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
st.session_state.selected_risks = selected_risks
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


# investor profile
# ▶️  right after st.sidebar.expander("⚙️  Settings"):
with st.sidebar.expander("🧑‍💼  Investor profile", expanded=True):
    experience_level   = st.radio("Experience",   ["Beginner", "Intermediate", "Expert"])
    explanation_pref   = st.radio("Detail level", ["Just the strategy", "Explain the reasoning", "Both"])

# 🔄  Store & show sticky pill
st.session_state.experience_level  = experience_level
st.session_state.explanation_pref  = explanation_pref
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
    # 1.  Build prompt ----------------------------------------------------------
    ignored = "; ".join(st.session_state.risk_ignore) or "None"
    total_capital = sum(st.session_state.portfolio_alloc.values())
    risk_string = ", ".join(risk_list) or "None"
    alloc_str = "; ".join(
        f"{k}: ${v:,.0f}" for k, v in st.session_state.portfolio_alloc.items()
    ) or "None provided"

    # Build user-style guidance from profile
    experience_note = {
        "Beginner": "Use simple, jargon-free language appropriate for a retail investor.",
        "Intermediate": "Use moderate technical terms and explain key terms when needed.",
        "Expert": "Use professional investment language without oversimplification.",
    }[st.session_state.experience_level]

    explanation_note = {
        "Just the strategy": "Skip all explanations — just give the hedge table and summary.",
        "Explain the reasoning": "For each hedge, explain the logic behind the choice.",
        "Both": "Include the full hedge table, and explain the rationale for each entry in clear terms.",
    }[st.session_state.explanation_pref]

    # Main prompt with guidance embedded
    prompt = textwrap.dedent(f"""
        Act as a **hedging strategist**.

        • **Basket**: {', '.join(basket)}
        • **Current allocation**: {alloc_str}
        • **Total capital**: ${total_capital:,.0f}
        • **Horizon**: {horizon} months
        • **Beta band**: {beta_rng[0]:.2f}–{beta_rng[1]:.2f}
        • **Stop-loss**: {stop_loss} %
        • **Detected headline risks** for {primary}: {risk_string}
        • **Ignore** the following risks: {ignored}

        ### Investor profile
        Experience level: {st.session_state.experience_level}
        Preferences: {st.session_state.explanation_pref}
        Style guidance: {experience_note} {explanation_note}

        ### Instructions:
        Design a tactical hedge to offset risk while preserving conviction positions.

        For each hedge, include 1–2 **tickers** (ETF, inverse, option proxy, or macro exposure).

        Return **only markdown**, in this exact format:

        1️⃣ A table with columns: **Ticker | Position | Amount ($) | Rationale | Source**  
        – Use a real clickable URL in the *Source* column.

        2️⃣ `### Summary`: a short paragraph (max 300 chars) summarizing the strategy.

        3️⃣ `### Residual Risks`: a numbered list (≤ 25 words each), each ending with a **source URL**.

        Do NOT wrap any output in code blocks or quotes.
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

            # Clean amounts
            df["Amount ($)"] = (
                df["Amount ($)"]
                .astype(str)
                .str.replace("$", "")
                .str.replace(",", "")
                .str.extract(r"(\d+\.?\d*)")[0]
                .astype(float)
            )

            df["Price"] = "_n/a_"
            df["Δ 1d %"] = "_n/a_"
            df["Source"] = "Suggested hedge"
            df = df[["Ticker", "Position", "Amount ($)", "Price", "Δ 1d %", "Rationale", "Source"]]

            # STEP 2: Combine with user's portfolio
            user_df = editor_df.copy()
            user_df["Position"] = "Long"
            user_df["Source"] = "User portfolio"
            user_df["Rationale"] = "—"
            user_df["Ticker"] = user_df["Ticker"].astype(str)
            user_df = user_df[["Ticker", "Position", "Amount ($)", "Price", "Δ 1d %", "Rationale", "Source"]]

            # STEP 3: Merge tables
            combined_df = pd.concat([user_df, df], ignore_index=True)

            st.dataframe(combined_df, use_container_width=True)

            st.markdown("### 📊 Post-Hedge Allocation Overview")
            pie_df = combined_df.copy()
            pie_df["Label"] = pie_df["Ticker"] + " (" + pie_df["Position"] + ")"
            pie_df["Amount"] = pie_df["Amount ($)"]
            st.plotly_chart(
                px.pie(
                    pie_df,
                    names="Label",
                    values="Amount",
                    title="Post-Hedge Portfolio",
                    hole=0.3
                ).update_traces(textinfo="label+percent"),
                use_container_width=True
            )

        except Exception as e:
            st.warning(f"Could not parse or merge hedge table: {e}")

        # 🔍 📌 Hedge Strategy Rationale (dynamic, with links)
        st.markdown("### 📌 Hedge Strategy Rationale")

        hedge_only = df[df["Source"] == "Suggested hedge"]

        if hedge_only.empty:
            st.info("No hedge rationale to show.")
        else:
            total_hedge = hedge_only["Amount ($)"].sum()
            st.markdown(
                f"A total of **${total_hedge:,.0f}** was allocated to hedge instruments to mitigate key risks in the portfolio.\n\n"
                "Below is the reasoning behind each hedge component:"
            )

            for _, row in hedge_only.iterrows():
                rationale = row.get("Rationale", "").strip()
                source = row.get("Source", "").strip()

                if re.match(r"^https?://", source):
                    st.markdown(f"- **{row['Ticker']}** → {rationale}  \n  ↪ [Source]({source})")
                else:
                    st.markdown(f"- **{row['Ticker']}** → {rationale}")

    # ───────────────────── PORTFOLIO vs HEDGE COMPOSITION ─────────────────────
    st.markdown("### 📊 Portfolio vs Hedge Allocation Breakdown")

    import io
    import plotly.graph_objects as go

    # Try to parse the markdown table from the plan
    md_lines = plan_md.splitlines()
    table_lines = [line for line in md_lines if '|' in line and not line.startswith('###')]

    if len(table_lines) >= 3:
        try:
            # STEP 1: Parse markdown table
            table_str = '\n'.join(table_lines)
            df = pd.read_csv(io.StringIO(table_str), sep='|')
            df.columns = [c.strip() for c in df.columns]
            df = df.dropna(subset=['Ticker', 'Amount ($)'])

            # Clean and parse amounts
            df["Amount ($)"] = (
                df["Amount ($)"]
                .astype(str)
                .str.replace("$", "")
                .str.replace(",", "")
                .str.extract(r"(\d+\.?\d*)")[0]
                .astype(float)
            )
            df["Price"] = "_n/a_"
            df["Δ 1d %"] = "_n/a_"
            df["Source"] = "Suggested hedge"

            # Reorder columns for consistency
            df = df[["Ticker", "Position", "Amount ($)", "Price", "Δ 1d %", "Rationale", "Source"]]

            # STEP 2: Extract user portfolio as DataFrame
            user_df = editor_df.copy()
            user_df["Position"] = "Long"
            user_df["Source"] = "User portfolio"
            user_df["Rationale"] = "—"
            user_df["Ticker"] = user_df["Ticker"].astype(str)
            user_df = user_df[["Ticker", "Position", "Amount ($)", "Price", "Δ 1d %", "Rationale", "Source"]]

            # STEP 3: Merge user + strategy dataframes
            combined_df = pd.concat([user_df, df], ignore_index=True)
            #st.markdown("---")  # Visually separates from earlier sections
            #st.markdown("### 🧾 Unified Portfolio + Strategy Table")
            #st.dataframe(combined_df, use_container_width=True)

            # STEP 4: Pie Chart – Post-Hedge Allocation
            pie_df = combined_df.copy()
            pie_df["Label"] = pie_df["Ticker"] + " (" + pie_df["Position"] + ")"
            pie_df["Amount"] = pie_df["Amount ($)"]

            with st.sidebar:
                if st.checkbox("📊 Show Post-Hedge Pie Chart", value=True, key="sidebar_post_hedge_pie"):
                    st.markdown("#### 🧮 Post-Hedge Allocation")
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
            st.warning(f"Could not render unified table or charts: {e}")


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
