import streamlit as st
from stock_utils import get_stock_summary
from openai_client import ask_openai

st.set_page_config(page_title="Stock Movement Chatbot", page_icon="📈")
st.title("📈 Market Movement Chatbot")
st.write("Enter a stock symbol and ask a question to analyze recent movement.")

ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA)", value="AAPL")

if ticker:
    summary = get_stock_summary(ticker)
    st.markdown(f"#### Stock Summary:\n{summary}")

    user_input = st.text_area("Ask a question about this stock", placeholder="e.g., Should I buy this stock now?")
    
    if st.button("Ask"):
        system_prompt = "You are a financial analyst assistant. Use the summary provided to answer the user's question."
        answer = ask_openai(system_prompt, f"Summary: {summary}\n\nQuestion: {user_input}")
        
        st.markdown("#### 🤖 Chatbot Answer")
        st.write(answer)
