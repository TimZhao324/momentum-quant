"""Compute standard performance and risk metrics across universe and benchmarks.

Calculates NAV growth, CAGR, annualized volatility, Sharpe ratio, and maximum drawdown.
Outputs:
    web/charts/02_nav.png         Cumulative NAV performance (log scale)
    web/charts/02_drawdown.png    Underwater drawdown trajectory
"""

import glob
import math
import os

# Hop to project root directory
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PANEL, LINE, TEXT, MUTED = "#171e2e", "#26304a", "#e7ecf5", "#8b96ad"
ACCENT, GRAY, GOOD, BAD = "#4ea1ff", "#7a869c", "#39d98a", "#ff6b6b"

# ===========================================================================
# 1) Ingest all CSV price files into consolidated daily table
# ===========================================================================
print("=== 1. Loading all CSV price histories ===")
columns = {}
for path in sorted(glob.glob("data/*.csv")):
    ticker = os.path.basename(path)[:-4]
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    columns[ticker] = df["Close"]

daily = pd.DataFrame(columns).sort_index()
print(f"Daily price matrix: {daily.shape} (days, tickers)")

# ===========================================================================
# 2) Resample daily prices to monthly frequency
# ===========================================================================
print("\n=== 2. Resampling to monthly frequency ===")
monthly = daily.resample("ME").last()

last_day = daily.index.max()
month_end = last_day + pd.offsets.MonthEnd(0)
if (month_end - last_day).days > 5:
    print(f"Data ends {last_day.date()} while {month_end.date()} is month end "
          "-> dropping incomplete trailing month")
    monthly = monthly.iloc[:-1]

print(f"Monthly price matrix: {monthly.shape} from {monthly.index[0].date()} to {monthly.index[-1].date()}")

# ===========================================================================
# 3) Core performance metrics: SPY benchmark calculation
# ===========================================================================
print("\n=== 3. SPY benchmark performance metrics ===")
spy = monthly["SPY"].pct_change(fill_method=None).dropna()
print(f"{len(spy)} monthly returns")

# NAV: Growth of $1
nav = [1.0]
for r in spy:
    nav.append(nav[-1] * (1 + r))
print(f"NAV        : $1 grew to ${nav[-1]:.2f}")

# CAGR: Compound Annual Growth Rate
n = len(spy)
cagr = (nav[-1] / nav[0]) ** (12 / n) - 1
print(f"CAGR       : {cagr:.2%} per year ({n} months = {n/12:.1f} years)")

# Volatility: Annualized return standard deviation
vol = spy.std() * math.sqrt(12)
print(f"Volatility : {vol:.2%} per year")

# Sharpe Ratio (Assuming zero risk-free rate)
sharpe = (spy.mean() / spy.std()) * math.sqrt(12)
print(f"Sharpe     : {sharpe:.2f} (assuming risk-free rate = 0)")

# Maximum Drawdown: Peak-to-trough decline
peak, max_dd, dd_curve = nav[0], 0.0, []
for v in nav:
    peak = max(peak, v)
    dd = v / peak - 1
    dd_curve.append(dd)
    max_dd = min(max_dd, dd)
print(f"Max DD     : {max_dd:.2%}")
print(f"Worst month: {spy.min():.2%}")
print(f"VaR 95%    : {spy.quantile(0.05):.2%}")

# ===========================================================================
# 4) Cross-asset comparison
# ===========================================================================
print("\n=== 4. Asset comparison scorecard ===")

def five_numbers(monthly_returns):
    """Calculate summary performance metrics for a return series."""
    s = pd.Series(list(monthly_returns), dtype="float64")
    line = [1.0]
    for r in s:
        line.append(line[-1] * (1 + r))
    peak, mdd = line[0], 0.0
    for v in line:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    return {
        "nav": line[-1],
        "cagr": (line[-1] / line[0]) ** (12 / len(s)) - 1,
        "vol": s.std() * math.sqrt(12),
        "sharpe": (s.mean() / s.std()) * math.sqrt(12),
        "mdd": mdd,
        "curve": line,
    }

SHOW = ["SPY", "AAPL", "NVDA", "KO", "INTC"]
print(f"{'':6} {'$1 ->':>9} {'CAGR':>9} {'vol':>8} {'Sharpe':>8} {'maxDD':>9}")
results = {}
for t in SHOW:
    m = five_numbers(monthly[t].pct_change(fill_method=None).dropna())
    results[t] = m
    print(f"{t:6} {m['nav']:>8.2f}x {m['cagr']:>8.2%} {m['vol']:>8.2%} "
          f"{m['sharpe']:>8.2f} {m['mdd']:>9.2%}")

# ===========================================================================
# 5) Generate charts
# ===========================================================================
print("\n=== 5. Plotting ===")
os.makedirs("web/charts", exist_ok=True)

def style(ax):
    """Apply standard chart styling."""
    ax.set_facecolor(PANEL)
    ax.grid(True, color=LINE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)

dates = monthly.index[1:]

# Chart 1: NAV Growth of $1
fig, ax = plt.subplots(figsize=(9, 4.6))
fig.patch.set_facecolor(PANEL)
style(ax)
colors = {"SPY": GRAY, "AAPL": ACCENT, "NVDA": GOOD, "KO": "#e6b23a", "INTC": BAD}
for t in SHOW:
    ax.plot(dates, results[t]["curve"][1:], label=t,
            color=colors[t], linewidth=2.0 if t == "SPY" else 1.3)
ax.set_yscale("log")
ax.set_yticks([1, 2, 5, 10, 20, 50])
ax.set_yticklabels(["$1", "$2", "$5", "$10", "$20", "$50"])
ax.set_title("Growth of $1, monthly, log scale")
ax.legend(facecolor=PANEL, edgecolor=LINE, labelcolor=TEXT, fontsize=9)
fig.savefig("web/charts/02_nav.png", dpi=150, facecolor=PANEL, bbox_inches="tight")
plt.close(fig)
print("   web/charts/02_nav.png")

# Chart 2: Underwater Drawdown
fig, ax = plt.subplots(figsize=(9, 3.4))
fig.patch.set_facecolor(PANEL)
style(ax)
ax.fill_between(dates, [d * 100 for d in dd_curve[1:]], 0, color=BAD, alpha=0.35)
ax.plot(dates, [d * 100 for d in dd_curve[1:]], color=BAD, linewidth=1.1)
ax.set_title(f"SPY drawdown — peak-to-trough decline (max: {max_dd:.1%})")
ax.set_ylabel("% below peak")
fig.savefig("web/charts/02_drawdown.png", dpi=150, facecolor=PANEL, bbox_inches="tight")
plt.close(fig)
print("   web/charts/02_drawdown.png")

# ===========================================================================
# CHECKPOINT
# ===========================================================================
print("\n=== CHECKPOINT ===")
assert daily.shape == (2895, 109), f"Expected (2895, 109), got {daily.shape}"
assert monthly.shape == (138, 109), f"Expected 138 months, got {monthly.shape[0]}"
assert str(monthly.index[-1].date()) == "2026-06-30", "Last month should be June 2026"
assert isinstance(spy, pd.Series), "SPY should be a Series"
assert abs(nav[-1] - 4.53) < 0.05, f"SPY $1 -> ${nav[-1]:.2f}, expected ~4.53"
assert abs(cagr - 0.1415) < 0.002, f"SPY CAGR {cagr:.4f}, expected ~0.1415"
assert 0.10 < vol < 0.30, f"Annual vol {vol:.3f} out of expected range"
assert abs(sharpe - 0.96) < 0.03, f"SPY Sharpe {sharpe:.2f}, expected ~0.96"
assert abs(max_dd - (-0.2393)) < 0.005, f"SPY maxDD {max_dd:.4f}, expected ~-0.2393"
print(f"OK: {len(monthly)} months, SPY $1 -> ${nav[-1]:.2f}, Sharpe {sharpe:.2f}, maxDD {max_dd:.1%}")
