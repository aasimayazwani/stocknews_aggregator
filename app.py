# app.py ─ Multi-ticker Market-Movement Chatbot  (autocomplete edition)
import streamlit as st
import requests, yfinance as yf
import pandas as pd
import textwrap
import re
import plotly.express as px
import math
from typing import Dict
import plotly.graph_objects as go   # for the confidence gauge
import numpy as np


from config import DEFAULT_MODEL
from stock_utils import get_stock_summary
from openai_client import ask_openai

st.set_page_config(page_title="Market Movement Chatbot", layout="wide")
# 👉 Drop the style block right here
st.markdown("""
<style>
.card{
  background:#1e1f24;
  padding:18px;
  border-radius:12px;
  margin-bottom:18px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.metric-tile {
  background:#f1f5f90D;  /* translucent */
  border:1px solid #33415550;
  padding:18px 22px;
  border-radius:12px;
  transition:background 0.2s;
  cursor:pointer;
}
.metric-tile:hover { background:#33415522; }
.metric-title  { font-weight:600; font-size:18px; margin-bottom:6px; }
.metric-value  { font-size:22px; font-weight:700; }
.chevron { float:right; font-size:20px; line-height:18px; transform:translateY(2px); }
</style>
""", unsafe_allow_html=True)

st.title("📈 Market Movement Chatbot")

# ───────────────────────── Session state ─────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "tickers_selected" not in st.session_state:
    st.session_state.tickers_selected = ["AAPL", "MSFT"]  # sensible defaults

def add_to_history(role, txt):
    st.session_state.history.append((role, txt))

def clean_llm_markdown(md: str) -> str:
    """Remove stray markdown emphasis and join-up words."""
    md = re.sub(r"(\d)(?=[a-zA-Z])", r"\1 ", md)     # 1.35reflects → 1.35 reflects
    md = re.sub(r"([a-zA-Z])(?=\d)", r"\1 ", md)     # at1.28       → at 1.28
    md = md.replace("*", "").replace("_", "")        # strip * and _
    return md

# ───────────────────────── KPI helpers ─────────────────────────
def quarters_sparkline(tkr: str, kpi: str = "revenue") -> go.Figure:
    """
    Tiny line chart for the last 8 quarters of Revenue or Net-Income.
    kpi ∈ {"revenue", "earnings"}
    """

    tk = yf.Ticker(tkr)

    # ---------- 1. pull the right financial dataframe ----------
    df = None

    # yfinance ≥ 0.2.33 exposes .quarterly_income_stmt  (preferred)
    if hasattr(tk, "quarterly_income_stmt"):
        df = tk.quarterly_income_stmt
    # legacy fallback
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        df = getattr(tk, "income_stmt", None)  # annual
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return go.Figure()  # give up gracefully

    # ---------- 2. map KPI ----------
    col_map = {
        "revenue": ["Total Revenue", "Revenue"],
        "earnings": ["Net Income", "Earnings"]
    }
    possible_rows = col_map.get(kpi.lower(), [])
    sel_row = next((r for r in possible_rows if r in df.index), None)
    if sel_row is None:
        return go.Figure()  # row not found

    ser = df.loc[sel_row].tail(8)  # last 8 quarters / years
    ser.index = ser.index.astype(str)

    # ---------- 3. make sparkline ----------
    fig = px.line(ser, height=110)
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
    )
    return fig

def kpi_bar(street: float, model: float, label: str) -> go.Figure:
    """Horizontal bar showing Street vs Model for one KPI."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[label], x=[street], name="Street", orientation="h",
        marker=dict(color="#334155")
    ))
    fig.add_trace(go.Bar(
        y=[label], x=[model], name="Model", orientation="h",
        marker=dict(color="#6366f1")
    ))
    fig.update_layout(
        barmode="group", height=140, margin=dict(l=60, r=20, t=10, b=10),
        showlegend=False, xaxis=dict(title="")
    )
    return fig


# ───────────────────────── Earnings helpers ──────────────────────────
def get_consensus_estimates(ticker: str) -> Dict[str, float]:
    """
    Return a dict with Street EPS and Revenue estimates for the *current* quarter.
    Falls back to NaN if Yahoo data missing.
    """
    tk = yf.Ticker(ticker)
    try:
        # yfinance >= 0.2.31 provides .earnings_forecasts
        eps = tk.earnings_forecasts.loc["eps_avg"][0]
    except Exception:
        eps = math.nan
    try:
        rev = tk.earnings_forecasts.loc["revenue_avg"][0] / 1e9  # convert to $M
    except Exception:
        rev = math.nan
    return {"EPS": eps, "Revenue": rev}


def extract_model_preds(outlook_md: str) -> Dict[str, float]:
    """
    Parse the LLM’s markdown and pull numeric model predictions for EPS & Revenue.
    We assume lines like '**EPS**: $4.80' or '**Total Revenue**: $60 100 M'.
    Expand the regex if you add more KPIs.
    """
    preds = {}
    # EPS
    eps_match = re.search(r"\bEPS.*?\$?([\d\.]+)", outlook_md, re.I)
    if eps_match:
        preds["EPS"] = float(eps_match.group(1).replace(",", ""))
    # Revenue
    rev_match = re.search(r"(?:Total )?Revenue.*?\$?([\d,\.]+)", outlook_md, re.I)
    if rev_match:
        preds["Revenue"] = float(rev_match.group(1).replace(",", ""))
    return preds


# ───────────────────────── Yahoo search helper ───────────────────
# ───────────────────────── Yahoo search helper ───────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def search_ticker_symbols(query: str, limit: int = 10):
    """Return [{'symbol': 'AAPL', 'name': 'Apple Inc.'}, …] for ‘query’."""
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "quotesCount": limit, "newsCount": 0, "lang": "en"}

    try:
        resp = requests.get(
            url,
            params=params,
            headers={"User-Agent": "Mozilla/5.0"},   # ←▸ important
            timeout=4,
        )
        resp.raise_for_status()
        quotes = resp.json().get("quotes", [])
    except Exception:
        # propagate “no matches” rather than cache a failure
        return []

    results = []
    for q in quotes:
        sym  = q.get("symbol", "").upper()
        name = q.get("shortname") or q.get("longname") or ""
        if sym and name:
            results.append({"symbol": sym, "name": name})
    return results
# ───────────────────────── Cached helpers ────────────────────────
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

# ───────────────────────── Sidebar settings ──────────────────────
with st.sidebar.expander("⚙️ Settings", expanded=False):
    model = st.selectbox(
        "OpenAI Model",
        #[DEFAULT_MODEL, "gpt-4.1-mini", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4o"],
        [DEFAULT_MODEL, "gpt-4.1-mini", "gpt-4o-mini"],
        0,
    )
    if st.button("🧹 Clear Chat History"):
        st.session_state.history = []
    if st.button("🛑 Clear Tickers"):
        st.session_state.tickers_selected = []
tickers = st.session_state.tickers_selected
# ───────────────────────── Ticker search UI ──────────────────────
with st.container():
    st.markdown("### 📌 Stock Selection", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([3, 2, 2])

    with col1:
        search_q = st.text_input("Search company or ticker", "", key="search_box")

    with col2:
        primary = st.selectbox("Primary ticker", options=tickers, index=0, key="primary_select")

    #with col3:
    #    domain_selected = st.selectbox("Domain", domains)

# Autocomplete below
if len(search_q) >= 2:
    matches = search_ticker_symbols(search_q)
    if not matches:
        st.info("No matches yet… keep typing")
    else:
        display_opts = [f"{m['name']}  ({m['symbol']})" for m in matches]
        choice = st.selectbox("Suggestions", display_opts, index=0, key="suggest_box")

        if st.button("➕ Add to basket", key="add_btn"):
            chosen_sym = choice.split("(")[-1].rstrip(")")
            default_seed = {"AAPL", "MSFT"}
            if set(st.session_state.tickers_selected) == default_seed:
                st.session_state.tickers_selected = []  # clear default demo list

            if chosen_sym not in st.session_state.tickers_selected:
                st.session_state.tickers_selected.insert(0, chosen_sym)  # insert at top
            st.rerun()

# Manual fallback (keeps parity with old flow)
#manual_raw = st.text_input("Or paste comma-separated symbols", "")
#if manual_raw:
##    for t in manual_raw.split(","):
#       sym = t.strip().upper()
#        if sym and sym not in st.session_state.tickers_selected:
#            st.session_state.tickers_selected.append(sym)

tickers = st.session_state.tickers_selected
if not tickers:
    st.info("Add at least one ticker to proceed.")
    st.stop()

#primary = tickers[0]  # first drives snapshot & sector
#primary = st.selectbox(
#    "Reference ticker (drives snapshot & peers)",
#    options=tickers,
#    index=0,
#    key="primary_select",
#)

# ───────────────────────── Snapshot & metadata ───────────────────
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
    st.markdown("### 🧾 Snapshot", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='text-align: center; font-size: 30px; font-weight: bold; color: white;'>
        ${last_px:.2f}
    </div>
    <div style='text-align: center; font-size: 16px; color: {"green" if pct_px >= 0 else "red"};'>
        {pct_px:+.2f}%
    </div>
    <hr style='margin:10px 0; border:1px solid #333;'/>
    <div style='font-size: 13px;'>Market Cap: <b>${info.get('marketCap', 0)/1e9:.2f} B</b></div>
    <div style='font-size: 13px;'>P/E Ratio: <b>{info.get('trailingPE', '—')}</b></div>
    """, unsafe_allow_html=True)

# ───────────────────────── Domain & competitor logic ─────────────
domains = [d for d in (sector, industry) if d] or ["General"]
domain_selected = st.selectbox("Domain context", domains)

if len(tickers) == 1:
    competitors_all = fetch_competitors_llm(model, primary, domain_selected)
    basket = [primary] + competitors_all[:3]
else:
    competitors_all = tickers[1:]
    basket = tickers

# ───────────────────────── Tabs ──────────────────────────────────
tab_compare, tab_strategy, tab_chat, tab_outlook = st.tabs(["📈 Compare", "🎯 Strategy", "💬 Chat", "🔮 Quarterly Outlook"])

# 1) Compare tab
with tab_compare:
    st.subheader("Price Comparison")
    #comps_selected = st.multiselect("Select symbols to plot",
    #                                options=basket + competitors_all,
    #                                default=basket)
    show_benchmark = st.checkbox("📊 Compare to S&P 500 (SPY)", value=True)

    comps_selected = st.multiselect("Select symbols to plot",
                                    options=basket + competitors_all,
                                    default=basket)

    # Add SPY unless already selected or explicitly unchecked
    if show_benchmark and "SPY" not in comps_selected:
        comps_selected.append("SPY")
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
        st.markdown("### 💹 Latest Prices", unsafe_allow_html=True)
        cols = st.columns(len(price_df.columns))

        for c, sym in zip(cols, price_df.columns):
            ser = price_df[sym]
            last = ser.iloc[-1]
            delta = ser.pct_change().iloc[-1] * 100

            with c:
                st.plotly_chart(px.line(ser, height=80)
                    .update_layout(showlegend=False, margin=dict(l=0,r=0,t=0,b=0),
                                xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False)),
                    use_container_width=True
                )
                st.markdown(f"""
                <div style='font-size: 20px; font-weight: bold;'>{sym}</div>
                <div style='font-size: 18px;'>${last:.2f}</div>
                <div style='color: {"green" if delta >= 0 else "red"};'>{delta:+.2f}%</div>
                """, unsafe_allow_html=True)


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
            f"Design a {goal.lower()} equity strategy using the basket [{basket_txt}]. "
            f"Sector focus: {sector_in}. Hedge or avoid exposure to {avoid_sym}. "
            f"Allocate a total of ${capital} over a {horizon}-month time horizon. "
            f"Match pair betas within the range {beta_rng[0]:.2f} to {beta_rng[1]:.2f}, "
            f"and apply a {stop_loss}% stop-loss to short positions. \n\n"
            "Return a markdown table with the following columns: Ticker, Position (Long/Short), "
            "Amount, and Rationale. Then provide a short paragraph summarizing the strategy. \n\n"
            "**Also include 2–3 current risk factors associated with this strategy. For each risk, "
            "cite the source explicitly (e.g., 'Source: Goldman Sachs Q2 Outlook, June 2025').**"
        )

        with st.spinner("Generating…"):
            plan = ask_openai(
                model,
                "You are a portfolio strategist. Output a table + narrative.",
                prompt,
            )

        st.markdown("### 📌 Suggested Strategy")
        st.write(plan)

        # Highlight risks
        def extract_risks_section(text: str):
            match = re.search(r"(### Risks.*?)(?=\n### |\Z)", text, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else None

        risks = extract_risks_section(plan)
        if risks:
            st.markdown("### ⚠️ Highlighted Risks")
            st.markdown(
                f"<div style='background-color:#2a2e35; padding:16px; border-radius:10px; color:#f1f5f9;'>"
                f"<pre style='white-space: pre-wrap; font-size: 14px;'>{risks}</pre></div>",
                unsafe_allow_html=True
            )
        else:
            st.info("No specific risks with sources were found in the strategy output.")


# 3) Chat tab
with tab_chat:
    for role, msg in st.session_state.history:
        st.chat_message(role).write(msg)
    q = st.chat_input("Ask anything…")
    if q:
        add_to_history("user", q)
        ctx = f"Summary: {summary}\nDomain: {domain_selected}\nTickers: {', '.join(basket)}"
        ans = ask_openai(model, "You are a helpful market analyst.", ctx + "\n\n" + q)
        add_to_history("assistant", ans)
        #st.experimental_rerun()
        st.rerun()

# ───────────────────────── Tab 4: Quarterly Outlook ─────────────────────────
# ───────────────────────── Tab 4: Quarterly Outlook ─────────────────────────
with tab_outlook:
    st.subheader("🔮 Quarterly Outlook: Consensus Intelligence")

    # 3.1  LLM prompt with numbers + prob
    outlook_prompt = (
        f"Provide numeric forecasts for **EPS** and **Total Revenue** for {primary}'s next quarter. "
        f"For each KPI include: your prediction, Street consensus, and beat probability in %. "
        f"Add one sentence of reasoning ending with 'Source: …'. "
        f"Return in markdown: a table plus bullets, no code fences."
    )

    with st.spinner("⏳ Generating outlook…"):
        outlook_md = ask_openai(
            model,
            "You are a senior equity analyst, precise and data-driven.",
            outlook_prompt,
        )

    # 3.2  Show the raw LLM text in a card
    #st.markdown("<div class='card'>", unsafe_allow_html=True)
    #st.write(outlook_md)
    #st.markdown("</div>", unsafe_allow_html=True)
    outlook_md_clean = clean_llm_markdown(outlook_md)

    st.markdown(
        textwrap.dedent(f"""
        <div class='card'>
        {outlook_md_clean}
        </div>
        """),
        unsafe_allow_html=True
    )

    # 3.3  Parse the numbers out of the LLM text
    def grab_num(pattern, text):
        m = re.search(pattern, text, re.I)
        if not m:
            return np.nan
        raw = m.group(1).replace(",", "")
        val = float(raw)
        # look ahead for optional “B”/“M”
        suffix_match = re.search(rf"{re.escape(raw)}\s*([BM])", text[m.end()-3:m.end()+2], re.I)
        if suffix_match and suffix_match.group(1).upper() == "M":
            val /= 1_000  # convert millions → billions
        return val

        # 3.3  Parse numbers FROM THE CLEAN STRING
    eps_model   = grab_num(r"EPS.*?\\$?([\\d\\.]+)", outlook_md_clean)
    rev_model   = grab_num(r"Revenue.*?\\$?([\\d\\.]+)", outlook_md_clean)
    eps_street  = grab_num(r"Street.*EPS.*?\\$?([\\d\\.]+)", outlook_md_clean)
    rev_street  = grab_num(r"Street.*Revenue.*?\\$?([\\d\\.]+)", outlook_md_clean)
    prob_match  = re.search(r"Probability.*?(\\d{{1,3}}) ?%", outlook_md_clean, re.I)
    prob_eps    = int(prob_match.group(1)) if prob_match else None

        # helper right above this block (or put it with your helpers section)
        # ── helper already defined just above ──────────────────────────
    
    def pct_delta(model_val, street_val):
        if np.isnan(model_val) or np.isnan(street_val) or street_val == 0:
            return ""
        return f"{(model_val - street_val) / street_val * 100:+.1f}% vs Street"

    # ── KPI interactive tiles (EPS & Revenue) ──────────────────────
    bullets = re.findall(r"•\s*(.+)", outlook_md_clean)
    while len(bullets) < 2:              # avoid IndexError
        bullets.append("Reason not found.")

    kpis = [
        dict(name="EPS",
             model=eps_model,
             street=eps_street,
             prob=prob_eps,
             spark_key="earnings",
             reason=bullets[0]),
        dict(name="Revenue",
             model=rev_model,
             street=rev_street,
             prob=None,
             spark_key="revenue",
             reason=bullets[1]),
    ]

    tiles = st.columns(2)                # two tiles per row
    for idx, k in enumerate(kpis):
        col        = tiles[idx % 2]
        state_key  = f"show_{k['name']}"
        if state_key not in st.session_state:
            st.session_state[state_key] = False

        tile_html = f"""
        <div class='metric-tile'>
            <span class='metric-title'>{k['name']}</span>
            <span class='metric-value'>${k['model']:,.2f}</span>
            <span class='chevron'>{'▼' if st.session_state[state_key] else '▶'}</span>
        </div>
        """
        if col.button(tile_html,
                      key=f"tile_{k['name']}",
                      use_container_width=True,
                      unsafe_allow_html=True):
            st.session_state[state_key] = not st.session_state[state_key]

        if st.session_state[state_key]:
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(
                    f"**Street consensus:** ${k['street']:,.2f}  \n"
                    f"**Δ vs Street:** {pct_delta(k['model'], k['street'])}"
                )
                st.plotly_chart(
                    quarters_sparkline(primary, k['spark_key']),
                    use_container_width=True,
                )
                if k['prob'] is not None:
                    st.plotly_chart(
                        go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=k['prob'],
                            gauge={"axis": {"range": [0, 100]}},
                        )),
                        use_container_width=True,
                    )
                st.write(k['reason'])
                st.markdown("</div>", unsafe_allow_html=True)

    # ───────────────────────── 8-Q Trend (leave as-is) ─────────────
    st.markdown("#### 🕒 8-Q Trend")
