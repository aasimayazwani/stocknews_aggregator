import re
import requests
import pandas as pd
import yfinance as yf
from typing import List
from config import DEFAULT_MODEL
from openai_client import ask_openai
import os
import streamlit as st

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