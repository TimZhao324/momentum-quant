"""Load equity price data, compute returns, and generate exploratory charts.

Usage:
    .venv/bin/python steps/01_load_and_plot.py

Outputs:
    web/charts/01_price.png       Historical adjusted close price series
    web/charts/01_histogram.png   Distribution of daily returns
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Hop to project root directory
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TICKER = "AAPL"

# ===========================================================================
# 1) Read price series from CSV
# ===========================================================================
print(f"=== 1. Reading data/{TICKER}.csv ===")
prices = pd.read_csv(f"data/{TICKER}.csv", index_col=0, parse_dates=True)

print(prices.head())
print("...")
print(f"shape   : {prices.shape}   <- (rows, columns) = (days, fields)")
print(f"columns : {list(prices.columns)}")
print(f"from {prices.index[0].date()} to {prices.index[-1].date()}")

# ===========================================================================
# 2) Series extraction and indexing
# ===========================================================================
print("\n=== 2. One column is a Series ===")
close = prices["Close"]

print(f"type    : {type(close).__name__}")
print(f"first 3 :\n{close.head(3)}")
print(f"\n.iloc[0]  (first row by position)    = {close.iloc[0]:.2f}")
print(f".iloc[-1] (last row by position)     = {close.iloc[-1]:.2f}")
print(f".loc['2020-03-23'] (by date index)   = {close.loc['2020-03-23']:.2f}")

# ===========================================================================
# 3) Vectorised return calculation
# ===========================================================================
print("\n=== 3. Daily returns: loop vs vectorised ===")
by_hand = []
for i in range(1, len(close)):
    yesterday = close.iloc[i - 1]
    today = close.iloc[i]
    by_hand.append(today / yesterday - 1)

vectorised = close.pct_change(fill_method=None).dropna()

biggest_gap = max(abs(a - b) for a, b in zip(by_hand, vectorised))
print(f"loop count       : {len(by_hand)} returns")
print(f"pct_change count : {len(vectorised)} returns")
print(f"max difference   : {biggest_gap:.2e}")
assert biggest_gap < 1e-10, "Loop and pct_change results must agree"

daily_returns = vectorised

print(f"\nbest day  : {daily_returns.max():+.2%} on {daily_returns.idxmax().date()}")
print(f"worst day : {daily_returns.min():+.2%} on {daily_returns.idxmin().date()}")
print(f"average   : {daily_returns.mean():+.4%} per day")

# ===========================================================================
# 4) Missing values and NaN behavior
# ===========================================================================
print("\n=== 4. Missing values ===")
raw = close.pct_change(fill_method=None)
print(f"pct_change returns {raw.isna().sum()} initial NaN value(s)")

# ===========================================================================
# 5) Generate charts
# ===========================================================================
print("\n=== 5. Plotting ===")
os.makedirs("web/charts", exist_ok=True)

PANEL, LINE, TEXT, MUTED, ACCENT = "#171e2e", "#26304a", "#e7ecf5", "#8b96ad", "#4ea1ff"

def style(ax):
    """Apply dark theme to matplotlib axis."""
    ax.set_facecolor(PANEL)
    ax.grid(True, color=LINE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)

# Chart 1: Price history
fig, ax = plt.subplots(figsize=(9, 4.2))
fig.patch.set_facecolor(PANEL)
style(ax)
ax.plot(close.index, close.values, color=ACCENT, linewidth=1.3)
ax.set_title(f"{TICKER} — daily close, split & dividend adjusted")
ax.set_ylabel("price (USD)")
fig.savefig("web/charts/01_price.png", dpi=150, facecolor=PANEL, bbox_inches="tight")
plt.close(fig)
print("   web/charts/01_price.png")

# Chart 2: Daily return distribution
fig, ax = plt.subplots(figsize=(9, 4.2))
fig.patch.set_facecolor(PANEL)
style(ax)
ax.hist(daily_returns.values * 100, bins=80, color=ACCENT, alpha=0.85)
ax.axvline(0, color=MUTED, linewidth=1)
ax.set_title(f"{TICKER} — distribution of daily returns (%)")
ax.set_xlabel("daily return (%)")
ax.set_ylabel("number of days")
fig.savefig("web/charts/01_histogram.png", dpi=150, facecolor=PANEL, bbox_inches="tight")
plt.close(fig)
print("   web/charts/01_histogram.png")

# ===========================================================================
# CHECKPOINT
# ===========================================================================
print("\n=== CHECKPOINT ===")
assert isinstance(prices.index, pd.DatetimeIndex), "Index must be a DatetimeIndex"
assert list(prices.columns) == ["Close", "High", "Low", "Open", "Volume"]
assert len(daily_returns) == len(prices) - 1, "Expected one return per day except first"
assert prices.shape[0] in (2895, 2769), f"Unexpected number of days: {prices.shape[0]}"
if TICKER == "AAPL":
    assert abs(close.iloc[0] - 24.1926) < 0.001, f"First AAPL close was {close.iloc[0]}"
print(f"OK: {len(prices)} days of {TICKER}, {len(daily_returns)} daily returns, 2 charts generated.")
