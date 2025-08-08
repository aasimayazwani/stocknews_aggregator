from __future__ import annotations

# ─────────────────────────────────────────────────────────────
# Streamlit must set page config BEFORE any other Streamlit call
# ─────────────────────────────────────────────────────────────
import streamlit as st
st.set_page_config(page_title="Hedge Strategy Chatbot", layout="wide")

# ─────────────────────────────────────────────────────────────
# Standard libs
# ─────────────────────────────────────────────────────────────
import pandas as pd
from io import StringIO
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Local imports
# ─────────────────────────────────────────────────────────────
from ui_components import render_strategy_cards, render_rationale, clean_md
# render_backtest_chart will be imported; if not present, we define a fallback below
try:
    from ui_components import render_backtest_chart  # type: ignore
except Exception:
    render_backtest_chart = None  # fallback defined later

from utils import search_tickers, fetch_prices, web_risk_scan, fetch_backtest_data
from strategy_generator import generate_strategies
from backtest import backtest_strategy
from config import DEFAULT_MODEL
from openai_client import ask_openai

# ─────────────────────────────────────────────────────────────
# Optional external CSS (guarded)
# ─────────────────────────────────────────────────────────────
css_path = Path(__file__).parent / "style.css"
try:
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
    else:
        # Non-fatal: just proceed with default styles
        pass
except Exception as _css_err:
    # Also non-fatal: do not block app if CSS read fails
    pass

# ─────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────
defaults = {
    "history": [],
    "portfolio": [],
    "portfolio_alloc": {},
    "alloc_df": None,
    "outlook_md": None,
    "risk_cache": {},
    "risk_ignore": [],
    "chosen_strategy": None,
    "strategy_legs": {},
    "strategy_history": [],
    "strategy_df": pd.DataFrame(),
    "backtest_duration": 12,
    "selected_strategy_idx": None,
    "prev_exp": None,
    "allowed_instruments": [],
    "experience_level": "Expert",
    "explanation_pref": "Just the strategy",
    "time_horizon": 6,
    # keep last computed total for backtest after reruns
    "last_total_capital": 0.0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────
# Sidebar: Investor Profile
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📌 Investor Profile")
    st.selectbox(
        label="Investor experience",
        options=["Beginner", "Intermediate", "Expert"],
        index=["Beginner", "Intermediate", "Expert"].index(st.session_state.get("experience_level", "Expert")),
        format_func=lambda x: f"Experience: {x}",
        key="experience_level"
    )
    st.selectbox(
        label="Explanation detail preference",
        options=["Just the strategy", "Explain the reasoning", "Both"],
        index=["Just the strategy", "Explain the reasoning", "Both"].index(st.session_state.get("explanation_pref", "Just the strategy")),
        format_func=lambda x: f"Detail level: {x}",
        key="explanation_pref"
    )
    st.slider("Time horizon (months):", 1, 24, st.session_state.get("time_horizon", 6), key="time_horizon")

    exp_map = {
        "Beginner": ["Inverse ETFs", "Commodities"],
        "Intermediate": ["Put Options", "Inverse ETFs", "Commodities"],
        "Expert": ["Put Options", "Collar Strategy", "Inverse ETFs", "Short Selling", "Volatility Hedges", "Commodities", "FX Hedges"],
    }
    all_instr = list({i for v in exp_map.values() for i in v})
    curr_exp = st.session_state.experience_level
    if st.session_state.get("prev_exp") != curr_exp:
        st.session_state.allowed_instruments = exp_map[curr_exp]
        st.session_state.prev_exp = curr_exp

    st.multiselect(
        "Allowed hedge instruments:",
        all_instr,
        default=st.session_state.allowed_instruments,
        key="allowed_instruments"
    )

    st.markdown("### 🧹 Session Tools")
    st.slider("Backtest period (months):", 1, 24, st.session_state.backtest_duration, key="backtest_duration")

    if st.button("🗂️ Reset Portfolio", key="clear_portfolio_btn"):
        for k in ["portfolio", "portfolio_alloc", "alloc_df"]:
            st.session_state[k] = [] if k != "alloc_df" else None
        st.session_state.last_total_capital = 0.0
        st.rerun()

    if st.button("💬 Clear Chat", key="clear_chat_btn"):
        st.session_state.history = []
        st.rerun()

    if st.button("🗑️ Clear Strategy History", key="clear_strategy_btn"):
        st.session_state.strategy_history = []
        st.rerun()

# ─────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────
st.title("Equity Strategy Assistant")

# ─────────────────────────────────────────────────────────────
# Portfolio
# ─────────────────────────────────────────────────────────────
st.subheader("📊 Portfolio")
with st.expander("Upload Portfolio (CSV)", expanded=True):
    up = st.file_uploader("Upload your portfolio", type=["csv"])
    if up:
        try:
            df = pd.read_csv(StringIO(up.getvalue().decode("utf-8")), engine="python", on_bad_lines="warn")
        except Exception as e:
            st.error("Error parsing CSV.")
            st.exception(e)
            st.stop()

        required_cols = {"Ticker", "Amount ($)"}
        if not required_cols.issubset(df.columns):
            st.error("CSV must contain 'Ticker' and 'Amount ($)' columns.")
            st.stop()

        df["Ticker"] = df["Ticker"].astype(str).str.upper()
        if "Stop-Loss ($)" not in df.columns:
            df["Stop-Loss ($)"] = None

        st.session_state.alloc_df = df[["Ticker", "Amount ($)", "Stop-Loss ($)"]]
        st.session_state.portfolio = df["Ticker"].tolist()
        st.session_state.portfolio_alloc = dict(zip(df["Ticker"], df["Amount ($)"]))

        prices = fetch_prices(st.session_state.portfolio, period="2d")
        disp = st.session_state.alloc_df.copy()
        if isinstance(prices, pd.DataFrame) and not prices.empty and len(prices.index) >= 2:
            last, prev = prices.iloc[-1], prices.iloc[-2]
            disp["Price"] = last.reindex(st.session_state.portfolio).round(2).values
            disp["Δ 1d %"] = ((last - prev) / prev * 100).reindex(st.session_state.portfolio).round(2).values
        else:
            disp["Price"], disp["Δ 1d %"] = 0.0, 0.0
        st.dataframe(disp, use_container_width=True)
    else:
        st.warning("Please upload a portfolio CSV to continue.")
        st.stop()

# ─────────────────────────────────────────────────────────────
# Strategy Generation
# ─────────────────────────────────────────────────────────────
if st.button("⚡ Generate", key="suggest_btn"):
    if not st.session_state.portfolio_alloc:
        st.warning("No portfolio — cannot suggest strategies.")
        st.stop()

    total = float(sum(st.session_state.portfolio_alloc.values()))
    st.session_state.last_total_capital = total

    # build risk string from cache if present
    risk_bits = []
    for ticker in st.session_state.portfolio:
        cached = st.session_state.risk_cache.get(ticker) or []
        # expected structure: List[Tuple[str, Any]] or List[List[str, Any]]
        for item in cached:
            try:
                risk = item[0]
                if isinstance(risk, str):
                    risk_bits.append(risk)
            except Exception:
                continue
    risks = "; ".join(risk_bits)

    strat_df = generate_strategies(
        model=DEFAULT_MODEL,
        portfolio=st.session_state.portfolio,
        total_capital=total,
        horizon=st.session_state.time_horizon,
        risk_string=risks,
        allowed_instruments=st.session_state.allowed_instruments,
        explanation_pref=st.session_state.explanation_pref,
        experience_level=st.session_state.experience_level
    )
    st.session_state.strategy_df = strat_df

    st.session_state.strategy_history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "horizon": st.session_state.time_horizon,
        "capital": total,
        "beta_band": (0.9, 1.1),
        "strategy_df": strat_df.copy() if isinstance(strat_df, pd.DataFrame) else pd.DataFrame(),
        "rationale_md": "\n\n".join(
            f"**{row['name']} ({row.get('variant', '')})**\n- {row['rationale'].get('thesis', '')}\n- {row['rationale'].get('tradeoff', '')}"
            for _, row in (strat_df.iterrows() if isinstance(strat_df, pd.DataFrame) else [])
            if isinstance(row.get('rationale', {}), dict)
        )
    })

if isinstance(st.session_state.strategy_df, pd.DataFrame) and not st.session_state.strategy_df.empty:
    st.subheader("🛡️ Recommended Hedging Strategies")
    render_strategy_cards(st.session_state.strategy_df)

# ─────────────────────────────────────────────────────────────
# Backtesting
# ─────────────────────────────────────────────────────────────
if st.session_state.chosen_strategy:
    st.info(f"Chosen: {st.session_state.chosen_strategy.get('name', '—')}")
    if st.button("📊 Run Backtest"):
        port = list(st.session_state.portfolio)
        i = st.session_state.selected_strategy_idx

        # infer hedge tickers from legs if available
        strategy_legs = st.session_state.strategy_legs.get(i, []) if isinstance(st.session_state.strategy_legs, dict) else []
        hedge = []
        for leg in strategy_legs:
            try:
                base = str(leg['instrument']).split()[0]
                hedge.append(base)
            except Exception:
                continue

        # fetch price history
        data = fetch_backtest_data(port + hedge, period=f"{st.session_state.backtest_duration}m")

        # determine total capital (recompute from alloc or use last known)
        if st.session_state.portfolio_alloc:
            total_bt = float(sum(st.session_state.portfolio_alloc.values()))
        else:
            total_bt = float(st.session_state.get("last_total_capital", 0.0))

        result = backtest_strategy(
            st.session_state.chosen_strategy,
            {t: data[t] for t in port if t in data},
            {h: data[h] for h in hedge if h in data},
            total_bt
        )

        st.subheader("Backtest Results")
        st.markdown(f"""
        - **Unhedged**: ${result.get('unhedged_final_value', 0.0):,.2f}
        - **Hedged**: ${result.get('hedged_final_value', 0.0):,.2f}
        - **Risk Reduction**: {result.get('risk_reduction_pct', 0.0):.2f}%
        - **Max Drawdown (Unhedged)**: {result.get('max_drawdown_unhedged', 0.0):.2f}%
        - **Max Drawdown (Hedged)**: {result.get('max_drawdown_hedged', 0.0):.2f}%
        - **Total Hedge Cost**: ${result.get('total_cost', 0.0):,.2f}
        """)

        # chart rendering (with fallback)
        if isinstance(result, dict) and all(k in result for k in ("unhedged_values", "hedged_values", "dates")):
            if callable(render_backtest_chart):
                render_backtest_chart(result["unhedged_values"], result["hedged_values"], result["dates"])
            else:
                # minimal fallback if UI component not available
                import pandas as pd  # local import to avoid top-level dep if not used
                df_chart = pd.DataFrame({
                    "date": result["dates"],
                    "unhedged": result["unhedged_values"],
                    "hedged": result["hedged_values"],
                }).set_index("date")
                st.line_chart(df_chart)

# ─────────────────────────────────────────────────────────────
# Quick Chat
# ─────────────────────────────────────────────────────────────
st.divider()
st.markdown("### 💬 Quick chat")
for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

if q := st.chat_input("Ask anything…"):
    st.session_state.history.append(("user", q))
    try:
        resp = ask_openai(
            DEFAULT_MODEL,
            "You are a helpful market analyst.",
            f"User portfolio: {','.join(st.session_state.portfolio)}\n{q}"
        )
    except Exception as e:
        resp = f"LLM request failed: {e}"
    st.session_state.history.append(("assistant", resp))
    st.rerun()