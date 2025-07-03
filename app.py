# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

from config import DEFAULT_MODEL          # your config file
from stock_utils import get_stock_summary # existing helper
from openai_client import ask_openai      # MUST accept (model, system, user)

# ───────────────────── Page & Global Setup ─────────────────────
st.set_page_config(
    page_title="Market Movement Chatbot",
    page_icon="📈",
    layout="wide",
)

# Persist chat history
if "history" not in st.session_state:
    st.session_state.history = []

def add_to_history(role: str, text: str):
    st.session_state.history.append((role, text))

# ───────────────────── Caching helpers ─────────────────────────
@st.cache_data(ttl=300)
def fetch_stock_df(tickers: list[str], period: str = "6mo") -> pd.DataFrame:
    """Historical close prices for the list of tickers."""
    df = yf.download(tickers, period=period, progress=False)["Close"]
    return df.dropna(axis=1, how="all")

@st.cache_data(ttl=300)
def fetch_competitors_llm(model: str, name: str, domain: str) -> list[str]:
    """
    Ask the LLM for competitor tickers in the specified domain.
    Returns a list of symbols (strings).
    """
    prompt = (
        f"You are a financial analyst. List the top 5 public companies "
        f"that compete with {name} in the “{domain}” domain. "
        f"Return ONLY the ticker symbols, as a Python list."
    )
    resp = ask_openai(model, "You are a helpful stock analyst.", prompt)

    # Attempt to evaluate a proper Python list
    try:
        tickers = eval(resp)
        if isinstance(tickers, list):
            return [t.strip().upper() for t in tickers]
    except Exception:
        pass  # fall through to naive parse

    # Fallback: first token per line
    return [
        line.strip().split()[0].strip('",[]').upper()
        for line in resp.splitlines() if line.strip()
    ]

# ───────────────────── Sidebar Settings ────────────────────────
with st.sidebar.expander("⚙️ Settings", expanded=False):
    model = st.selectbox(
        "OpenAI Model",
        options=[
            DEFAULT_MODEL,
            "gpt-4.1-mini",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4o",
        ],
        index=0,
    )
    if st.button("🧹 Clear Chat History"):
        st.session_state.history = []

# ───────────────────── Ticker Input ────────────────────────────
ticker = (
    st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA)", "AAPL")
    .upper()
    .strip()
)

if not ticker:
    st.stop()

# ───────────────────── Core Data Fetching ──────────────────────
summary = get_stock_summary(ticker)
add_to_history("bot", summary)  # first message shown in chat

# Get company info for domain selection
info = yf.Ticker(ticker).info or {}
company_name = info.get("longName", ticker)
sector   = info.get("sector")
industry = info.get("industry")
domains  = [d for d in (sector, industry) if d] or ["General"]

# Let user pick domain
domain_selected = st.selectbox("Which domain would you like to explore?", domains)

# Competitors via LLM
competitors = fetch_competitors_llm(model, company_name, domain_selected)

# ───────────────────── Layout: Tabs ────────────────────────────
tab_summary, tab_charts, tab_comp, tab_chat = st.tabs(
    ["📊 Summary", "📈 Charts", "🤝 Competitors", "💬 Chat"]
)

# ---------- Summary Tab ----------
with tab_summary:
    st.subheader(f"{ticker} Quick Stats")
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        latest = hist["Close"].iloc[-1]
        pct = (latest - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2] * 100
    except Exception:
        latest, pct = float("nan"), float("nan")

    col1, col2, col3 = st.columns(3)
    col1.metric("Price", f"${latest:.2f}", f"{pct:+.2f}%")
    col2.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.1f} B", "")
    col3.metric("P/E Ratio", str(info.get("trailingPE", "—")), "")

    st.markdown("##### LLM Stock Summary")
    st.write(summary)

# ---------- Charts Tab ----------
with tab_charts:
    st.subheader("Price Chart (6 months)")
    try:
        price_df = fetch_stock_df([ticker] + competitors)
        col_main, col_comp = st.columns(2)
        with col_main:
            fig_main = px.line(price_df[[ticker]], title=ticker)
            st.plotly_chart(fig_main, use_container_width=True)
        with col_comp:
            if competitors:
                fig_comp = px.line(price_df[competitors], title="Competitors")
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.info("No competitor price data to display.")
    except Exception as e:
        st.error(f"Chart error: {e}")

# ---------- Competitors Tab ----------
with tab_comp:
    st.subheader(f"Top Competitors in {domain_selected}")
    if competitors:
        cols = st.columns(len(competitors))
        for c, sym in zip(cols, competitors):
            try:
                data = yf.download(sym, period="1mo", progress=False)["Close"]
                c.metric(
                    sym,
                    f"${data.iloc[-1]:.2f}",
                    f"{(data.pct_change().iloc[-1]*100):+.2f}%",
                )
                spark = px.line(data, height=80, width=150)
                c.plotly_chart(
                    spark,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            except Exception:
                c.metric(sym, "—", "—")
    else:
        st.info("No competitors returned by the model.")

# ---------- Chat Tab ----------
with tab_chat:
    # Display the conversation
    for role, msg in st.session_state.history:
        st.chat_message(role).write(msg)

    # Chat input
    user_prompt = st.chat_input("Ask a question…")
    if user_prompt:
        add_to_history("user", user_prompt)

        system_prompt = "You are a financial analyst assistant. Use the context below."
        context = (
            f"Summary: {summary}\n"
            f"Competitors: {', '.join(competitors)}\n"
            f"Domain: {domain_selected}\n"
        )
        with st.spinner("Thinking…"):
            answer = ask_openai(model, system_prompt, context + "\n\n" + user_prompt)
        add_to_history("assistant", answer)
        st.experimental_rerun()  # refresh chat tab
