"""Core simulation engine for strategy backtesting.

Implements cross-sectional portfolio construction, transaction cost modeling,
short-borrow financing, and forward return attribution.
"""

import numpy as np
import pandas as pd

import config


def pick_weights(scores, mode):
    """Convert cross-sectional signal scores to target portfolio weights.

    Args:
        scores (pd.Series): Signal scores across constituent universe.
        mode (str): 'long_only' (top-N equal weight) or 'long_short' (dollar-neutral deciles).

    Returns:
        dict: Mapping of ticker to portfolio weight.
    """
    clean = scores.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) == 0:
        return {}
    ranked = clean.sort_values(ascending=False)

    if mode == "long_only":
        names = list(ranked.index[:config.TOP_N])
        return {t: 1.0 / len(names) for t in names}

    size = max(1, len(ranked) // config.DECILE)
    weights = {t: 0.5 / size for t in ranked.index[:size]}
    for t in ranked.index[-size:]:
        weights[t] = weights.get(t, 0.0) - 0.5 / size
    return weights


def backtest(px, vol, dates, score_fn, mode="long_only", borrow_bps=None):
    """Execute monthly rebalanced backtest over specified sample window.

    Accounts for 1-month look-ahead avoidance, transaction costs proportional to turnover,
    and annualized borrow costs on short notionals.
    """
    borrow_bps = config.BORROW_BPS_PER_YEAR if borrow_bps is None else borrow_bps
    cost_rate = (config.COMMISSION_BPS + config.SLIPPAGE_BPS) / 10000.0
    monthly_returns = px.pct_change(fill_method=None)

    gross, net, turnovers, earned_on = [], [], [], []
    previous = {}

    for i in range(config.WARMUP, len(dates) - 1):
        weights = pick_weights(score_fn(px, vol, i), mode)

        turnover = sum(abs(weights.get(t, 0.0) - previous.get(t, 0.0))
                       for t in set(weights) | set(previous))

        next_month = monthly_returns.iloc[i + 1]
        g = sum(w * next_month.get(t, 0.0)
                for t, w in weights.items() if pd.notna(next_month.get(t)))

        short_notional = sum(-w for w in weights.values() if w < 0)
        borrow_cost = short_notional * borrow_bps / 10000.0 / 12.0

        gross.append(g)
        net.append(g - cost_rate * turnover - borrow_cost)
        turnovers.append(turnover)
        earned_on.append(dates[i + 1])
        previous = weights

    holdings = pick_weights(score_fn(px, vol, len(dates) - 1), mode)
    return {
        "gross": gross,
        "net": net,
        "turnover": turnovers,
        "dates": earned_on,
        "avg_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "cost_drag_bps": float((np.mean(gross) - np.mean(net)) * 12 * 10000) if net else 0.0,
        "holdings": holdings,
    }


def benchmark_returns(px, dates, symbol=None):
    """Retrieve benchmark returns matching the exact backtest rebalancing window."""
    symbol = symbol or config.BENCHMARK
    r = px[symbol].pct_change(fill_method=None)
    return [float(r.iloc[i + 1]) for i in range(config.WARMUP, len(dates) - 1)]
