import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import time

from config import DEFAULT_MODEL
from stock_utils import get_stock_summary
from openai_client import ask_openai

st.set_page_config(page_title="Market Movement Chatbot", layout="wide")
st.title("📈 Market Movement Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

def add_to_history(role: str, text: str):
    st.session_state.history.append((role, text))

@st.cache_data(ttl=300)
def fetch_stock_df(tickers: list[str], period: str) -> pd.DataFrame:
    df = yf.download(tickers, period=period, progress=False)["Close"]
    return df.dropna(axis=1, how="all")

@st.cache_data(ttl=300)
def fetch_competitors_llm(model: str, name: str, domain: str) -> list[str]:
    prompt = (
        f"List ONLY the top 7 stock ticker symbols of companies that compete with {name} "
        f"in the '{domain}' domain. Return a plain Python list like ['MSFT', 'GOOG']."
    )
    response = ask_openai(model, "You are a helpful stock analyst.", prompt)
    try:
        tickers = eval(response.strip(), {"__builtins__": None}, {})
        return [t.strip().upper() for t in tickers if isinstance(t, str)]
    except:
        return [line.strip().split()[0].strip('",[]').upper()
                for line in response.splitlines() if line.strip()]

# ───────────── Sidebar ─────────────
with st.sidebar.expander("⚙️ Settings", expanded=False):
    model = st.selectbox("OpenAI Model", [
        DEFAULT_MODEL, "gpt-4.1-mini", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4o"
    ])
    if st.button("🧹 Clear Chat"):
        st.session_state.history = []

# ───────────── Stock Input ─────────────
ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA)", "AAPL").upper().strip()
if not ticker:
    st.stop()

summary = get_stock_summary(ticker)
add_to_history("bot", summary)

try:
    info = yf.Ticker(ticker).info
    company_name = info.get("longName", ticker)
    sector = info.get("sector")
    industry = info.get("industry")
except:
    info, company_name, sector, industry = {}, ticker, "Technology", None

domains = [d for d in (sector, industry) if d] or ["General"]
domain_selected = st.selectbox("Which domain would you like to explore?", domains)
all_competitors = fetch_competitors_llm(model, company_name, domain_selected)

# ───────────── Tabs ─────────────
tab_summary, tab_compare, tab_strategy, tab_chat = st.tabs(
    ["📊 Summary", "📈 Compare", "🎯 Strategy", "💬 Chat"]
)

# ───────────── Summary Tab ─────────────
with tab_summary:
    st.subheader(f"{ticker} Quick Stats")
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        latest = hist["Close"].iloc[-1]
        pct = (latest - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2] * 100
    except:
        latest, pct = float("nan"), float("nan")

    col1, col2, col3 = st.columns(3)
    col1.metric("Price", f"${latest:.2f}", f"{pct:+.2f}%")
    col2.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.1f} B", "")
    col3.metric("P/E Ratio", str(info.get("trailingPE", "—")), "")

    st.markdown("##### LLM Stock Summary")
    st.write(summary)

# ───────────── Compare Tab ─────────────
with tab_compare:
    st.subheader("📈 Compare Price Movement")

    selected_comps = st.multiselect("Select competitors to compare", options=all_competitors, default=all_competitors[:3])
    duration = st.selectbox("Select comparison duration", ["1mo", "3mo", "6mo", "1y"], index=2)

    compare_symbols = [ticker] + selected_comps
    df = fetch_stock_df(compare_symbols, duration)
    valid_symbols = df.columns.tolist()
    if ticker not in valid_symbols:
        st.warning(f"{ticker} has no price data in {duration} range.")
        df = df.drop(columns=[ticker], errors="ignore")
        ticker = None

    if not df.empty:
        st.plotly_chart(
            px.line(df, title=f"Price Comparison ({duration})", labels={"value": "Price", "variable": "Ticker"}),
            use_container_width=True
        )
    else:
        st.error("No valid price data found for selected symbols.")

    st.markdown("### 💹 Latest Prices and Change")
    valid_comps = [sym for sym in compare_symbols if sym in df.columns]
    cols = st.columns(len(valid_comps))
    for c, sym in zip(cols, valid_comps):
        try:
            sub = df[sym]
            latest = sub.iloc[-1]
            delta = sub.pct_change().iloc[-1] * 100
            spark = px.line(sub, height=80)
            spark.update_layout(
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False)
            )
            c.plotly_chart(spark, use_container_width=True)
            c.metric(sym, f"${latest:.2f}", f"{delta:+.2f}%")
        except:
            c.metric(sym, "—", "—")

# ───────────── Strategy Tab ─────────────
with tab_strategy:
    st.subheader("🎯 Strategy Assistant")

    sector_interest = st.text_input("Sector you're interested in", placeholder="e.g., EV, AI, Semiconductors")
    goal = st.selectbox("What is your positioning goal?", ["Long", "Short", "Hedged", "Neutral"])
    #concern = st.text_input("Any stock to hedge against or avoid?", placeholder="e.g., TSLA")
    default_concern = ticker if ticker else ""
    concern = st.text_input(
        "Any stock to hedge against or avoid?",
        value=default_concern,
        placeholder="e.g., TSLA"
    )
    if st.button("Suggest Strategy"):
        user_intent = f"""
        I want a strategy in the {sector_interest} sector.
        I am interested in a {goal.lower()} position.
        I want to hedge against or avoid: {concern if concern else 'none'}.
        Suggest a set of 2-3 stock or ETF positions I can take to express this view.
        Output a strategy with rationale.
        """
        with st.spinner("Analyzing strategy..."):
            strategy_response = ask_openai(
                model,
                "You are a portfolio strategist. Give thoughtful long/short ideas using public stocks or ETFs.",
                user_intent,
            )
        st.markdown("### 📌 Suggested Strategy")
        st.write(strategy_response)

# ───────────── Chat Tab ─────────────
with tab_chat:
    for role, text in st.session_state.history:
        st.chat_message(role).write(text)

    user_prompt = st.chat_input("Ask a question…")
    if user_prompt:
        add_to_history("user", user_prompt)
        context = (
            f"Summary: {summary}\n"
            f"Domain: {domain_selected}\n"
            f"Competitors: {', '.join(all_competitors)}\n"
        )
        system = "You are a helpful financial analyst assistant. Use the context below."
        with st.spinner("Thinking…"):
            answer = ask_openai(model, system, context + "\n\n" + user_prompt)
        add_to_history("assistant", answer)
        st.experimental_rerun()
