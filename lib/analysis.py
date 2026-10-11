"""Quantitative signal evaluation and factor risk decomposition.

Implements cross-sectional Information Coefficient (Rank IC),
quintile bucket return monotonicity testing, Ordinary Least Squares (OLS),
and Variance Inflation Factor (VIF) diagnostics.
"""

import math

import numpy as np
import pandas as pd

import config


def ic_series(px, vol, dates, score_fn, method="rank", min_names=None):
    """Compute monthly cross-sectional Information Coefficient series.

    Measures correlation between period-t signal scores and period-(t+1) forward returns.
    """
    min_names = config.MIN_NAMES_FOR_IC if min_names is None else min_names
    out = []
    for i in range(config.WARMUP, len(dates) - 1):
        scores = score_fn(px, vol, i)
        forward = px.iloc[i + 1] / px.iloc[i] - 1
        both = pd.concat([scores, forward], axis=1).dropna()
        if len(both) < min_names:
            continue
        a, b = both.iloc[:, 0], both.iloc[:, 1]
        if method == "rank":
            a, b = a.rank(), b.rank()
        out.append(float(a.corr(b)))
    return out


def ic_summary(series):
    """Compute summary statistics for an Information Coefficient series."""
    if not series:
        return {k: None for k in ("mean", "std", "ir", "t", "p", "hit_rate", "months")}
    T = len(series)
    mean = float(np.mean(series))
    std = float(np.std(series, ddof=1))
    ir = mean / std if std > 0 else 0.0
    t = ir * math.sqrt(T)
    return {
        "mean": mean,
        "std": std,
        "ir": ir,
        "t": t,
        "p": math.erfc(abs(t) / math.sqrt(2)),
        "hit_rate": sum(1 for x in series if x > 0) / T,
        "months": T,
    }


def quintile_buckets(px, vol, dates, score_fn, n_buckets=5):
    """Partition universe into score quantiles and track forward bucket returns."""
    buckets = [[] for _ in range(n_buckets)]
    for i in range(config.WARMUP, len(dates) - 1):
        scores = score_fn(px, vol, i)
        forward = px.iloc[i + 1] / px.iloc[i] - 1
        both = pd.concat([scores, forward], axis=1).dropna()
        both.columns = ["score", "forward"]
        both = both.sort_values("score", ascending=False)
        size = len(both) // n_buckets
        for b in range(n_buckets):
            seg = both["forward"].iloc[b * size:(b + 1) * size] if b < n_buckets - 1 \
                else both["forward"].iloc[(n_buckets - 1) * size:]
            buckets[b].append(float(seg.mean()))
    return buckets


def ols(y, factors):
    """Fit multi-factor linear regression: y = alpha + sum(beta_k * factor_k) + eps."""
    names = list(factors)
    y = np.asarray(y, dtype=float)
    n = len(y)

    X = np.column_stack([np.ones(n)] + [np.asarray(factors[k], dtype=float) for k in names])
    k = X.shape[1]
    assert X[:, 0].min() == 1.0, "Column 0 must be constant intercept"

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    dof = n - k

    s2 = float(resid @ resid / dof)
    se = np.sqrt(s2 * np.diag(np.linalg.inv(X.T @ X)))
    t = beta / se
    p = np.array([math.erfc(abs(v) / math.sqrt(2)) for v in t])

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot

    return {
        "names": ["ALPHA"] + names,
        "beta": beta,
        "se": se,
        "t": t,
        "p": p,
        "r2": r2,
        "adj_r2": 1 - (1 - r2) * (n - 1) / dof,
        "resid": resid,
        "fitted": fitted,
        "n": n,
        "k": k,
        "dof": dof,
        "resid_vol_ann": math.sqrt(s2) * math.sqrt(12),
    }


def vif(factors):
    """Compute Variance Inflation Factor (VIF) for multicollinearity diagnostics."""
    return {c: 1.0 / (1.0 - ols(factors[c], {k: v for k, v in factors.items() if k != c})["r2"])
            for c in factors}


def factor_returns(monthly_returns, dates, symbol, subtract=None):
    """Construct factor return series aligned to backtest periods."""
    window = range(config.WARMUP, len(dates) - 1)
    out = np.array([float(monthly_returns[symbol].iloc[i + 1]) for i in window])
    if subtract:
        out = out - np.array([float(monthly_returns[subtract].iloc[i + 1]) for i in window])
    return out


def build_factors(monthly_returns, dates):
    """Generate risk factor returns specified in config.REGRESSION_FACTORS."""
    return {name: factor_returns(monthly_returns, dates, sym, minus)
            for name, sym, minus in config.REGRESSION_FACTORS}
