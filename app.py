# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

from config import DEFAULT_MODEL
from stock_utils import get_stock_summary
from openai_client import ask_openai

st.set_page_config(page_title="Market Movement Chatbot", layout="wide")
st.title("📈 Market Movement Chatbot")

# ——————————————————————————————————————————————————————————————
# Session State: history of (role, text)
if "history" not in st.session_state:
    st.session_state.history = []

def add_to_history(role: str, text: str):
    st.session_state.history.append((role, text))

# ——————————————————————————————————————————————————————————————
# Caching data fetches
@st.cache_data(ttl=300)
def fetch_stock_df(tickers: list[str], period: str = "6mo") -> pd.DataFrame:
    """Download historical close prices for a list of tickers."""
    df = yf.download(tickers, period=period)["Close"]
    df = df.dropna(axis=1, how="all")  # drop symbols with no data
    return df

@st.cache_data(ttl=300)
def fetch_competitors_llm(model: str, name: str, domain: str) -> list[str]:
    """Ask the LLM for a competitor list, return tickers only."""
    prompt = (
        f"You are a financial analyst. List the top 5 public companies "
        f"that compete with {name} in the “{domain}” domain. "
        f"Return only the ticker symbols, as a Python list."
    )
    resp = ask_openai(model, "You are a helpful stock analyst.", prompt)
    # assume the LLM returns something like: ["MSFT", "GOOG", "AMZN", ...]
    try:
        return eval(resp)
    except:
        # fallback: split lines
        return [line.strip().split()[0].strip('",[]') for line in resp.splitlines()]

# ——————————————————————————————————————————————————————————————
# Sidebar: model selector & clear history
with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "Model",
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
    if st.button("Clear Chat History"):
        st.session_state.history = []

# ——————————————————————————————————————————————————————————————
# 1) Ticker input
ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA)", "AAPL") \
             .upper().strip()
if not ticker:
    st.stop()

# 2) Summary
summary = get_stock_summary(ticker)
st.markdown(f"#### Stock Summary:\n{summary}")
add_to_history("bot", summary)

# 3) Fetch sector/industry
info = yf.Ticker(ticker).info or {}
name     = info.get("longName", ticker)
sector   = info.get("sector")
industry = info.get("industry")
domains  = [d for d in (sector, industry) if d]

if domains:
    domain = st.selectbox("Which domain to explore?", domains)
    comps = fetch_competitors_llm(model, name, domain)
    st.markdown("#### 🔍 Competitors in this domain:")
    st.write(", ".join(comps))
    add_to_history("bot", f"Competitors in {domain}: {', '.join(comps)}")
else:
    st.info("No sector/industry info available.")
    comps = []

# 4) Price chart
if comps:
    df = fetch_stock_df([ticker] + comps)
    st.markdown("#### 📊 Price Chart (6 mo)")
    # two-column layout
    col1, col2 = st.columns(2)
    with col1:
        fig_main = px.line(df[[ticker]], title=ticker)
        st.plotly_chart(fig_main, use_container_width=True)
    with col2:
        fig_comp = px.line(df.drop(columns=[ticker]), title="Competitors")
        st.plotly_chart(fig_comp, use_container_width=True)

# 5) Free-form Q&A
st.markdown("#### 💬 Chat")
for role, text in st.session_state.history:
    if role == "user":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Bot:** {text}")

user_q = st.text_area("Your question", "")
if st.button("Ask"):
    add_to_history("user", user_q)
    system = "You are a financial analyst assistant. Use the summary to answer."
    user_msg = f"Summary: {summary}\nCompetitors: {comps}\n\nQuestion: {user_q}"
    with st.spinner("Thinking…"):
        ans = ask_openai(model, system, user_msg)
    add_to_history("bot", ans)
    st.experimental_rerun()
