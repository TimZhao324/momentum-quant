"""Common utilities, data ingestion, financial metrics, and plotting routines.
"""

import glob
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config

# Visual palette matching the web dashboard
BG     = "#0f1420"
PANEL  = "#171e2e"
LINE   = "#26304a"
TEXT   = "#e7ecf5"
MUTED  = "#8b96ad"
ACCENT = "#4ea1ff"
GRAY   = "#7a869c"
GOOD   = "#39d98a"
WARN   = "#e6b23a"
BAD    = "#ff6b6b"


def load_prices(folder=None, column="Close"):
    """Load specified column from all price CSVs into a single DataFrame.

    Returns:
        pd.DataFrame: Date-indexed DataFrame with tickers as columns.
    """
    folder = folder or config.DATA_FOLDER
    out = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        ticker = os.path.basename(path)[:-4]
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        out[ticker] = df[column]
    return pd.DataFrame(out).sort_index()


def to_monthly(daily, how="last"):
    """Resample daily prices to monthly frequency, dropping partial trailing months."""
    monthly = daily.resample("ME").last() if how == "last" else daily.resample("ME").mean()
    last_day = daily.index.max()
    month_end = last_day + pd.offsets.MonthEnd(0)
    if (month_end - last_day).days > 5:
        monthly = monthly.iloc[:-1]
    return monthly


def returns(prices):
    """Compute period percentage returns without forward filling gaps."""
    return prices.pct_change(fill_method=None)


def nav(period_returns):
    """Compute normalized growth of $1 from a return series."""
    line = [1.0]
    for r in period_returns:
        line.append(line[-1] * (1.0 + r))
    return line


def metrics(monthly_returns):
    """Compute performance scorecard from monthly returns.

    Assumes zero risk-free rate (raw returns).
    """
    s = pd.Series(list(monthly_returns), dtype="float64")
    n = len(s)
    if n < 2:
        return {k: None for k in ("total_return", "cagr", "vol", "sharpe",
                                  "max_drawdown", "worst_month", "down_month_rate",
                                  "var_95", "months")}
    line = nav(s)
    total = line[-1] / line[0] - 1.0
    cagr = ((line[-1] / line[0]) ** (12.0 / n) - 1.0) if line[-1] > 0 else None
    vol = float(s.std() * math.sqrt(12))
    sharpe = float((s.mean() / s.std()) * math.sqrt(12)) if s.std() > 0 else 0.0

    peak, mdd = line[0], 0.0
    for v in line:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1.0)
    return {
        "total_return": total,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "worst_month": float(s.min()),
        "down_month_rate": float((s < 0).mean()),
        "var_95": float(s.quantile(0.05)),
        "months": n,
    }


def beta_of(series, market):
    """Calculate market beta via cov(series, market) / var(market)."""
    a, b = pd.Series(list(series)), pd.Series(list(market))
    if b.var() == 0:
        return None
    return float(a.cov(b) / b.var())


def new_fig(width=9, height=4.2):
    """Initialize a styled matplotlib figure matching dashboard theme."""
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(PANEL)
    style_axes(ax)
    return fig, ax


def style_axes(ax):
    """Apply dashboard styling to matplotlib axes."""
    ax.set_facecolor(PANEL)
    ax.grid(True, color=LINE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    return ax


def save_fig(fig, name, folder=None):
    """Save figure to destination folder and close plot."""
    folder = folder or config.CHARTS_FOLDER
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name + ".png")
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print("   chart ->", path)
    return path


def pct(x, digits=2):
    """Format float as percentage string."""
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"
