import yfinance as yf

def get_stock_summary(ticker_symbol: str) -> str:
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="5d")

        if hist.empty:
            return f"No recent data found for {ticker_symbol.upper()}."

        latest_close = hist["Close"].iloc[-1]
        previous_close = hist["Close"].iloc[-2]
        change = latest_close - previous_close
        pct_change = (change / previous_close) * 100

        sentiment = "up" if change > 0 else "down" if change < 0 else "flat"
        return f"{ticker_symbol.upper()} is {sentiment}: {latest_close:.2f} USD ({pct_change:+.2f}%)."

    except Exception as e:
        return f"Error fetching stock data for {ticker_symbol}: {e}"
