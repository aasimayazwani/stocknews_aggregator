import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
import streamlit as st

def backtest_strategy(strategy: Dict, portfolio_data: Dict[str, pd.DataFrame], hedge_data: Dict[str, pd.DataFrame], capital: float) -> Dict:
    """
    Backtest a single hedging strategy using historical data.
    Args:
        strategy: Strategy dictionary with name, legs, and aggregate_cost_pct.
        portfolio_data: Historical price data for portfolio tickers.
        hedge_data: Historical price data for hedge instruments.
        capital: Total portfolio capital.
    Returns:
        Dictionary with backtest metrics (e.g., returns, drawdown, risk_reduction) and time series data.
    """
    # Initialize metrics and data validation
    if not portfolio_data or not any(df.empty for df in portfolio_data.values()):
        st.warning("No valid portfolio data available for backtesting.")
        return {
            'unhedged_final_value': capital,
            'hedged_final_value': capital,
            'risk_reduction_pct': 0.0,
            'max_drawdown_unhedged': 0.0,
            'max_drawdown_hedged': 0.0,
            'total_cost': strategy.get('aggregate_cost_pct', 0) * capital / 100,
            'dates': [],
            'unhedged_values': [capital],
            'hedged_values': [capital]
        }

    # Align dates across all dataframes
    common_dates = set.intersection(*(set(df.index) for df in portfolio_data.values()) | (set(df.index) for df in hedge_data.values()))
    if not common_dates:
        st.warning("No common dates found between portfolio and hedge data.")
        return {
            'unhedged_final_value': capital,
            'hedged_final_value': capital,
            'risk_reduction_pct': 0.0,
            'max_drawdown_unhedged': 0.0,
            'max_drawdown_hedged': 0.0,
            'total_cost': strategy.get('aggregate_cost_pct', 0) * capital / 100,
            'dates': [],
            'unhedged_values': [capital],
            'hedged_values': [capital]
        }

    # Get portfolio weights from session state (approximated if not available)
    portfolio_alloc = st.session_state.get('portfolio_alloc', {ticker: capital/len(portfolio_data) for ticker in portfolio_data})
    total_alloc = sum(portfolio_alloc.values())
    weights = {ticker: alloc / total_alloc for ticker, alloc in portfolio_alloc.items()}

    # Simulate unhedged portfolio
    unhedged_values = [capital]
    dates = []
    for date in sorted(common_dates):
        daily_return = np.mean([
            (portfolio_data[ticker].loc[date, 'Close'] / portfolio_data[ticker].loc[date, 'Open'] - 1) * weights.get(ticker, 1/len(portfolio_data))
            for ticker in portfolio_data
            if date in portfolio_data[ticker].index
        ])
        unhedged_values.append(unhedged_values[-1] * (1 + daily_return))
        dates.append(date)

    # Simulate hedged portfolio
    costs = strategy.get('aggregate_cost_pct', 0) * capital / 100
    hedged_values = [capital - costs]
    for date in dates[1:]:  # Skip initial date since cost is applied upfront
        daily_portfolio_return = np.mean([
            (portfolio_data[ticker].loc[date, 'Close'] / portfolio_data[ticker].loc[date, 'Open'] - 1) * weights.get(ticker, 1/len(portfolio_data))
            for ticker in portfolio_data
            if date in portfolio_data[ticker].index
        ])
        hedge_return = 0
        for leg in strategy.get('legs', []):
            instrument = leg['instrument']
            position = leg['position'].lower()
            notional_pct = leg['notional_pct'] / 100
            ticker = instrument.split()[0]  # Extract ticker (e.g., 'SPY' from 'Put Option on SPY')
            if ticker in hedge_data and date in hedge_data[ticker].index:
                hedge_price = hedge_data[ticker].loc[date, 'Close']
                hedge_prev = hedge_data[ticker].loc[date - timedelta(days=1), 'Close'] if (date - timedelta(days=1)) in hedge_data[ticker].index else hedge_price
                leg_return = (hedge_price / hedge_prev - 1) if hedge_prev != 0 else 0
                # Simple approximation for options (e.g., put option as inverse return)
                if 'option' in instrument.lower() and position == 'long':
                    leg_return = -leg_return  # Approximate put option effect
                elif position == 'short':
                    leg_return = -leg_return
                hedge_return += leg_return * notional_pct
        hedged_values.append(hedged_values[-1] * (1 + daily_portfolio_return + hedge_return))

    # Calculate metrics
    unhedged_values = np.array(unhedged_values)
    hedged_values = np.array(hedged_values)
    drawdown_unhedged = np.min(unhedged_values / np.maximum.accumulate(unhedged_values)) - 1
    drawdown_hedged = np.min(hedged_values / np.maximum.accumulate(hedged_values)) - 1
    risk_reduction = (drawdown_unhedged - drawdown_hedged) / abs(drawdown_unhedged) * 100 if drawdown_unhedged != 0 else 0

    return {
        'strategy_name': strategy['name'],
        'unhedged_final_value': unhedged_values[-1],
        'hedged_final_value': hedged_values[-1],
        'risk_reduction_pct': round(risk_reduction, 2),
        'max_drawdown_unhedged': round(drawdown_unhedged * 100, 2),
        'max_drawdown_hedged': round(drawdown_hedged * 100, 2),
        'total_cost': costs,
        'dates': [d.strftime('%Y-%m-%d') for d in dates],
        'unhedged_values': unhedged_values.tolist(),
        'hedged_values': hedged_values.tolist()
    }