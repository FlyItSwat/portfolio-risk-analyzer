"""Portfolio analytics on adjusted, split/dividend-aware daily price series."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def normalize_weights(weights):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) == 0 or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('Weights must be finite, nonnegative, and sum to more than zero.')
    return w / w.sum()


def daily_returns(prices):
    if not isinstance(prices, pd.DataFrame) or prices.shape[1] == 0:
        raise ValueError('Prices must be a DataFrame with at least one asset.')
    clean = prices.apply(pd.to_numeric, errors='coerce').dropna(how='any')
    if len(clean) < 3 or (clean <= 0).any().any():
        raise ValueError('At least three positive, aligned daily price observations are required.')
    returns = clean.pct_change(fill_method=None).dropna(how='any')
    if len(returns) < 2:
        raise ValueError('Not enough daily returns.')
    return returns


def portfolio_series(returns, weights, rebalance='daily'):
    """Daily rebalanced or buy-and-hold portfolio returns, assuming zero trading costs."""
    w = normalize_weights(weights)
    if len(w) != returns.shape[1]:
        raise ValueError('Number of weights must match the assets.')
    if rebalance == 'daily':
        return returns.dot(w).rename('Portfolio')
    if rebalance == 'buy-and-hold':
        relatives = (1 + returns).cumprod()
        wealth = relatives.dot(w)
        wealth = pd.concat([pd.Series([1.0]), wealth.reset_index(drop=True)], ignore_index=True)
        values = wealth.pct_change().iloc[1:].to_numpy()
        return pd.Series(values, index=returns.index, name='Portfolio')
    raise ValueError('Unknown rebalance mode.')


def metrics(returns, annual_rf_percent=0.0):
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) < 2 or (r <= -1).any():
        raise ValueError('At least two valid daily returns above -100% are required.')
    wealth = (1 + r).cumprod()
    total_return = float(wealth.iloc[-1] - 1)
    annual_return = float(wealth.iloc[-1] ** (TRADING_DAYS / len(r)) - 1)
    volatility = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    daily_rf = (1 + annual_rf_percent / 100) ** (1 / TRADING_DAYS) - 1 if annual_rf_percent > -100 else None
    if daily_rf is None:
        raise ValueError('Risk-free rate must exceed -100%.')
    sharpe = float((r.mean() - daily_rf) / r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.std(ddof=1) > 0 else float('nan')
    drawdown = wealth / pd.concat([pd.Series([1.0]), wealth]).cummax().iloc[1:].to_numpy() - 1
    return {'total_return': total_return, 'annualized_return': annual_return,
            'annualized_volatility': volatility, 'sharpe_ratio': sharpe,
            'max_drawdown': float(drawdown.min())}


def growth_index(returns, starting_value=10000):
    if starting_value <= 0:
        raise ValueError('Starting value must be positive.')
    return (1 + returns).cumprod() * starting_value
