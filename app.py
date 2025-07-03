import streamlit as st
import yfinance as yf

from stock_utils import get_stock_summary
from openai_client import ask_openai

st.set_page_config(page_title="Stock Movement Chatbot", page_icon="📈")
st.title("📈 Market Movement Chatbot")

# 1) User enters ticker
ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA)", value="AAPL").upper().strip()

if ticker:
    # 2) Get price summary
    summary = get_stock_summary(ticker)
    st.markdown(f"#### Stock Summary:\n{summary}")

    # 3) Fetch company info from yfinance
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        company_name = info.get("longName", ticker)
        sector       = info.get("sector", None)
        industry     = info.get("industry", None)
    except Exception:
        company_name, sector, industry = ticker, None, None

    # 4) Domain selection
    domains = []
    if sector:   domains.append(sector)
    if industry: domains.append(industry)
    if domains:
        domain_choice = st.selectbox("Which domain would you like to explore?", domains)
        
        # 5) Generate competitors via LLM
        if domain_choice:
            competitor_prompt = (
                f"You are a financial analyst. "
                f"List the top 5 public companies that compete with {company_name} "
                f"in the \"{domain_choice}\" domain. "
                f"Return them as a simple numbered list of ticker symbols and names."
            )
            with st.spinner("Fetching competitors…"):
                competitors = ask_openai("You are a helpful stock market assistant.", competitor_prompt)
            st.markdown("#### 🔍 Competitors in this domain:")
            st.write(competitors)
    else:
        st.info("No sector/industry info available for this ticker.")

    # 6) Free‐form Q&A
    user_input = st.text_area("Ask a question about this stock", placeholder="e.g., Should I buy this stock now?")
    if st.button("Ask"):
        system_prompt = "You are a financial analyst assistant. Use the summary provided to answer the user's question."
        full_prompt    = f"Summary: {summary}\n\nQuestion: {user_input}"
        with st.spinner("Thinking…"):
            answer = ask_openai(system_prompt, full_prompt)
        st.markdown("#### 🤖 Chatbot Answer")
        st.write(answer)
