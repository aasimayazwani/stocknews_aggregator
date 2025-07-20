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
        Dictionary with backtest metrics (e.g., returns, drawdown, risk_reduction).
    """
    # Initialize metrics
    portfolio_returns = []
    hedged_returns = []
    costs = strategy.get('aggregate_cost_pct', 0) * capital / 100
    horizon_months = strategy.get('horizon_months', 6)
    
    # Simulate portfolio performance (unhedged)
    portfolio_value = capital
    for date, prices in list(portfolio_data.values())[0].iterrows():
        # Simple equal-weighted portfolio return
        daily_return = np.mean([prices['Close'] / prices['Open'] - 1 for prices in portfolio_data.values()])
        portfolio_value *= (1 + daily_return)
        portfolio_returns.append(portfolio_value)
    
    # Simulate hedged performance
    hedged_value = capital - costs  # Subtract initial hedge cost
    for date, prices in list(portfolio_data.values())[0].iterrows():
        daily_portfolio_return = np.mean([prices['Close'] / prices['Open'] - 1 for prices in portfolio_data.values()])
        hedge_return = 0
        for leg in strategy.get('legs', []):
            instrument = leg['instrument']
            position = leg['position'].lower()
            notional_pct = leg['notional_pct'] / 100
            ticker = instrument.split()[0]  # Extract ticker (e.g., 'SPY' from 'Put Option on SPY')
            if ticker in hedge_data:
                hedge_price = hedge_data[ticker].loc[date, 'Close'] if date in hedge_data[ticker].index else 0
                hedge_prev = hedge_data[ticker].loc[:date].iloc[-2]['Close'] if len(hedge_data[ticker].loc[:date]) > 1 else hedge_price
                leg_return = (hedge_price / hedge_prev - 1) if hedge_prev != 0 else 0
                if position == 'short':
                    leg_return = -leg_return
                hedge_return += leg_return * notional_pct
        hedged_value *= (1 + daily_portfolio_return + hedge_return)
        hedged_returns.append(hedged_value)
    
    # Calculate metrics
    portfolio_returns = np.array(portfolio_returns)
    hedged_returns = np.array(hedged_returns)
    drawdown_unhedged = np.min(portfolio_returns / np.maximum.accumulate(portfolio_returns)) - 1
    drawdown_hedged = np.min(hedged_returns / np.maximum.accumulate(hedged_returns)) - 1
    risk_reduction = (drawdown_unhedged - drawdown_hedged) / abs(drawdown_unhedged) * 100 if drawdown_unhedged != 0 else 0
    
    return {
        'strategy_name': strategy['name'],
        'unhedged_final_value': portfolio_returns[-1],
        'hedged_final_value': hedged_returns[-1],
        'risk_reduction_pct': round(risk_reduction, 2),
        'max_drawdown_unhedged': round(drawdown_unhedged * 100, 2),
        'max_drawdown_hedged': round(drawdown_hedged * 100, 2),
        'total_cost': costs
    }