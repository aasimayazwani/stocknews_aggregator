import time
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

from config import DEFAULT_MODEL
from stock_utils import get_stock_summary
from openai_client import ask_openai

st.set_page_config(page_title="Market Movement Chatbot", layout="wide")
st.title("📈 Market Movement Chatbot")

# ─────────────── Session State for Chat ───────────────
if "history" not in st.session_state:
    st.session_state.history = []

def add_to_history(role: str, text: str):
    st.session_state.history.append((role, text))

# ─────────────── Cached Helpers ───────────────
@st.cache_data(ttl=300)
def fetch_stock_df(tickers: list[str], period: str = "6mo") -> pd.DataFrame:
    df = yf.download(tickers, period=period, progress=False)["Close"]
    return df.dropna(axis=1, how="all")

@st.cache_data(ttl=300)
def fetch_competitors_llm(model: str, name: str, domain: str) -> list[str]:
    prompt = (
        f"You are a financial analyst. List only the top 5 stock ticker symbols "
        f"of public companies that compete with {name} in the '{domain}' domain. "
        f"Return a plain Python list like ['MSFT', 'GOOG', 'NVDA']."
    )
    response = ask_openai(model, "You are a helpful stock analyst.", prompt)

    try:
        tickers = eval(response.strip(), {"__builtins__": None}, {})
        return [t.strip().upper() for t in tickers if isinstance(t, str)]
    except:
        tickers = []
        for line in response.splitlines():
            line = line.strip()
            if line and line.isalpha() and len(line) <= 5:
                tickers.append(line.upper())
        return tickers[:5]

# ─────────────── Sidebar Settings ───────────────
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

# ─────────────── Ticker Input ───────────────
ticker = (
    st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA)", "AAPL")
    .upper()
    .strip()
)

if not ticker:
    st.stop()

summary = get_stock_summary(ticker)
add_to_history("bot", summary)

info = yf.Ticker(ticker).info or {}
company_name = info.get("longName", ticker)
sector = info.get("sector")
industry = info.get("industry")
domains = [d for d in (sector, industry) if d] or ["General"]
domain_selected = st.selectbox("Which domain would you like to explore?", domains)

# ─────────────── Get Competitors ───────────────
competitors = fetch_competitors_llm(model, company_name, domain_selected)

# ─────────────── Filter Valid Tickers ───────────────
all_symbols = [ticker] + competitors
price_df = fetch_stock_df(all_symbols)
valid_symbols = price_df.columns.tolist()
competitors = [sym for sym in competitors if sym in valid_symbols]
ticker = ticker if ticker in valid_symbols else None

# ─────────────── Layout Tabs ───────────────
tab_summary, tab_charts, tab_comp, tab_chat = st.tabs(
    ["📊 Summary", "📈 Charts", "🤝 Competitors", "💬 Chat"]
)

# ─────────────── Summary Tab ───────────────
with tab_summary:
    st.subheader(f"{ticker} Quick Stats" if ticker else "No valid price data")
    if ticker:
        hist = yf.Ticker(ticker).history(period="5d")
        latest = hist["Close"].iloc[-1]
        pct = (latest - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2] * 100
        col1, col2, col3 = st.columns(3)
        col1.metric("Price", f"${latest:.2f}", f"{pct:+.2f}%")
        col2.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.1f} B", "")
        col3.metric("P/E Ratio", str(info.get("trailingPE", "—")), "")
    st.markdown("##### LLM Stock Summary")
    st.write(summary)

# ─────────────── Charts Tab ───────────────
with tab_charts:
    st.subheader("Price Chart (6 months)")
    if ticker:
        col_main, col_comp = st.columns(2)
        with col_main:
            fig_main = px.line(price_df[[ticker]], title=ticker)
            st.plotly_chart(fig_main, use_container_width=True)
        with col_comp:
            if competitors:
                fig_comp = px.line(price_df[competitors], title="Competitors")
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.info("No valid competitor prices.")
    else:
        st.error("No valid historical data for main ticker.")

# ─────────────── Competitors Tab ───────────────
with tab_comp:
    st.subheader(f"Top Competitors in {domain_selected}")
    if competitors:
        cols = st.columns(len(competitors))
        for c, sym in zip(cols, competitors):
            try:
                # SAFER one-by-one fetch:
                time.sleep(0.1)
                data = yf.Ticker(sym).history(period="1mo")
                if data.empty:
                    raise ValueError("No data returned")
                last_price = data["Close"].iloc[-1]
                delta = data["Close"].pct_change().iloc[-1] * 100

                c.metric(sym, f"${last_price:.2f}", f"{delta:+.2f}%")
                spark = px.line(data["Close"], height=80, width=150)
                c.plotly_chart(
                    spark,
                    use_container_width=True,
                    config={"displayModeBar": False}
                )
            except Exception as e:
                c.metric(sym, "—", "—")
    else:
        st.error("❌ No valid competitors found.")

# ─────────────── Chat Tab ───────────────
with tab_chat:
    for role, text in st.session_state.history:
        st.chat_message(role).write(text)

    user_prompt = st.chat_input("Ask a question…")
    if user_prompt:
        add_to_history("user", user_prompt)

        context = (
            f"Summary: {summary}\n"
            f"Domain: {domain_selected}\n"
            f"Competitors: {', '.join(competitors)}\n"
        )
        system = "You are a helpful financial analyst assistant. Use the context below."
        with st.spinner("Thinking…"):
            answer = ask_openai(model, system, context + "\n\n" + user_prompt)
        add_to_history("assistant", answer)
        st.experimental_rerun()
