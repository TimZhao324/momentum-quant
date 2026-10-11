"""Backtesting engine execution across long-only and dollar-neutral regimes.

Demonstrates look-ahead avoidance, transaction cost drag, market beta attribution,
and performance differentials between long-only and long-short implementations.

Outputs:
    web/charts/03_nav.png        Cumulative NAV vs benchmark (log scale)
    web/charts/03_turnover.png   Monthly turnover bar chart
    web/charts/03_scatter.png    Beta scatter plot (long-only vs long-short against SPY)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "lib")]
os.chdir(ROOT)

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import timlib as tl
import engine
from strategies import STRATEGIES, BY_ID

# ===========================================================================
# 1) Ingest data using shared library
# ===========================================================================
print("=== 1. Data Ingestion ===")
daily = tl.load_prices()
px = tl.to_monthly(daily)
vol = tl.to_monthly(tl.load_prices(column="Volume"), how="mean")
dates = list(px.index)
print(f"{len(dates)} months, {px.shape[1]} tickers, {dates[0].date()} .. {dates[-1].date()}")

spy = engine.benchmark_returns(px, dates)
print(f"Backtest sample window: {len(spy)} months")

# ===========================================================================
# 2) Cross-sectional scoring and portfolio construction sample
# ===========================================================================
print("\n=== 2. Cross-Sectional Scoring & Portfolio Weights ===")
mom = BY_ID["MOM12_1"]
i = 60
scores = mom["score"](px, vol, i)
print(f"Sample month {i} = {dates[i].date()}")
print(f"Universe scored: {scores.notna().sum()} tickers; Top 5 ranked:")
for t, s in scores.dropna().sort_values(ascending=False).head(5).items():
    print(f"   {t:6} {s:+.1%}")

w_long = engine.pick_weights(scores, "long_only")
w_ls = engine.pick_weights(scores, "long_short")
print(f"\nLong-only weights sum  : {sum(w_long.values()):.2f} across {len(w_long)} assets")
print(f"Long-short weights sum : {sum(w_ls.values()):+.2f} (dollar-neutral)")
print(f"   Long bucket  : {[t for t, x in w_ls.items() if x > 0][:5]} ...")
print(f"   Short bucket : {[t for t, x in w_ls.items() if x < 0][:5]} ...")

# ===========================================================================
# 3) Look-ahead bias demonstration
# ===========================================================================
print("\n=== 3. Look-Ahead Bias Demonstration ===")

def backtest_offset(score_fn, offset):
    """Execute simplified backtest with variable execution lag.

    offset=1: Honest execution (rebalance at t, earn forward return t+1).
    offset=0: Look-ahead contaminated execution (rebalance using t, earn return t).
    """
    out = []
    r = px.pct_change(fill_method=None)
    for idx in range(config.WARMUP, len(dates) - 1):
        w = engine.pick_weights(score_fn(px, vol, idx), "long_only")
        nxt = r.iloc[idx + offset]
        out.append(sum(x * nxt.get(t, 0.0) for t, x in w.items() if pd.notna(nxt.get(t))))
    return out

# Unlagged momentum: includes current month's return in ranking
mom12_0 = lambda p, v, idx: p.iloc[idx] / p.iloc[idx - 12] - 1

honest_net = backtest_offset(mom12_0, 1)
cheat_net = backtest_offset(mom12_0, 0)
honest_12_0, cheat = tl.metrics(honest_net), tl.metrics(cheat_net)

print(f"{'12-month momentum, unlagged':30}{'Sharpe':>8}{'CAGR':>10}{'$1 becomes':>14}")
print(f"{'  offset=1  honest':30}{honest_12_0['sharpe']:>8.2f}{tl.pct(honest_12_0['cagr']):>10}"
      f"{'$' + format(tl.nav(honest_net)[-1], ',.2f'):>14}")
print(f"{'  offset=0  look-ahead bias':30}{cheat['sharpe']:>8.2f}{tl.pct(cheat['cagr']):>10}"
      f"{'$' + format(tl.nav(cheat_net)[-1], ',.2f'):>14}")

assert cheat["sharpe"] > 2.5, "Contaminated backtest should produce artificially inflated Sharpe"
assert honest_12_0["sharpe"] < 2.0, "Realistic backtest Sharpe should remain bounded"

honest_raw = engine.backtest(px, vol, dates, mom["score"], mode="long_only")
honest = tl.metrics(honest_raw["net"])

# ===========================================================================
# 4) Long-only execution
# ===========================================================================
print("\n=== 4. Long-Only Execution: Top-N Portfolio ===")
lo = honest_raw
lo_m = honest
spy_m = tl.metrics(spy)
lo_beta = tl.beta_of(lo["net"], spy)

print(f"{'':14}{'$1 ->':>9}{'CAGR':>9}{'vol':>9}{'Sharpe':>8}{'maxDD':>9}{'beta':>7}{'turn':>7}")
print(f"{'strategy':14}{tl.nav(lo['net'])[-1]:>8.2f}x{tl.pct(lo_m['cagr']):>9}{tl.pct(lo_m['vol']):>9}"
      f"{lo_m['sharpe']:>8.2f}{tl.pct(lo_m['max_drawdown']):>9}{lo_beta:>7.2f}{lo['avg_turnover']:>7.2f}")
print(f"{'SPY':14}{tl.nav(spy)[-1]:>8.2f}x{tl.pct(spy_m['cagr']):>9}{tl.pct(spy_m['vol']):>9}"
      f"{spy_m['sharpe']:>8.2f}{tl.pct(spy_m['max_drawdown']):>9}{1.00:>7.2f}{'-':>7}")
print(f"\nTrading friction drag: {lo['cost_drag_bps']:.0f} bps/year "
      f"({lo['cost_drag_bps']/100:.2f}% annualized)")

active = np.array(lo["net"]) - np.array(spy)
te = active.std(ddof=1) * math.sqrt(12)
portfolio_ir = (active.mean() * 12) / te
t_stat = portfolio_ir * math.sqrt(len(active) / 12)
print(f"Active return: {active.mean()*12:.2%}/yr | Tracking error: {te:.2%} | "
      f"Information Ratio: {portfolio_ir:.2f} (t = {t_stat:.2f})")

# ===========================================================================
# 5) Long-short execution: Market beta neutralisation
# ===========================================================================
print("\n=== 5. Dollar-Neutral Long-Short Execution ===")

ls0 = engine.backtest(px, vol, dates, mom["score"], mode="long_short", borrow_bps=0)
ls0_m = tl.metrics(ls0["net"])
ls_beta = tl.beta_of(ls0["net"], spy)

print(f"{'':14}{'CAGR':>9}{'vol':>9}{'Sharpe':>8}{'maxDD':>9}{'beta to SPY':>13}")
print(f"{'long-only':14}{tl.pct(lo_m['cagr']):>9}{tl.pct(lo_m['vol']):>9}{lo_m['sharpe']:>8.2f}"
      f"{tl.pct(lo_m['max_drawdown']):>9}{lo_beta:>13.2f}")
print(f"{'long-short':14}{tl.pct(ls0_m['cagr']):>9}{tl.pct(ls0_m['vol']):>9}{ls0_m['sharpe']:>8.2f}"
      f"{tl.pct(ls0_m['max_drawdown']):>9}{ls_beta:>13.2f}")

print("\nBorrow fee sensitivity analysis:")
print(f"{'borrow cost':>14}{'CAGR':>9}{'Sharpe':>8}")
for bps in (0, 100, 300):
    m = tl.metrics(engine.backtest(px, vol, dates, mom["score"],
                                   mode="long_short", borrow_bps=bps)["net"])
    print(f"{str(bps) + ' bps/yr':>14}{tl.pct(m['cagr']):>9}{m['sharpe']:>8.2f}")

# ===========================================================================
# 6) Multi-strategy cross-comparison
# ===========================================================================
print("\n=== 6. Multi-Strategy Performance Comparison ===")
print(f"{'id':10}{'sleeve':12}{'LO Sharpe':>11}{'LO CAGR':>10}{'LS Sharpe':>11}{'LS CAGR':>10}{'turn':>7}")
table = {}
for s in STRATEGIES:
    a = engine.backtest(px, vol, dates, s["score"], mode="long_only")
    b = engine.backtest(px, vol, dates, s["score"], mode="long_short")
    ma, mb = tl.metrics(a["net"]), tl.metrics(b["net"])
    table[s["id"]] = (a, ma, b, mb)
    print(f"{s['id']:10}{s['sleeve']:12}{ma['sharpe']:>11.2f}{tl.pct(ma['cagr']):>10}"
          f"{mb['sharpe']:>11.2f}{tl.pct(mb['cagr']):>10}{a['avg_turnover']:>7.2f}")

# ===========================================================================
# 7) Visualizations
# ===========================================================================
print("\n=== 7. Generating Visualizations ===")
chart_dates = lo["dates"]

# Chart 1: Cumulative NAV
fig, ax = tl.new_fig(9, 4.6)
ax.plot(chart_dates, tl.nav(lo["net"])[1:], color=tl.ACCENT, linewidth=1.6, label="Momentum 12-1, long-only")
ax.plot(chart_dates, tl.nav(ls0["net"])[1:], color=tl.GOOD, linewidth=1.4, label="Momentum 12-1, long-short")
ax.plot(chart_dates, tl.nav(spy)[1:], color=tl.GRAY, linewidth=1.8, label="SPY")
ax.set_yscale("log")
ax.set_yticks([0.5, 1, 2, 5, 10, 20])
ax.set_yticklabels(["$0.5", "$1", "$2", "$5", "$10", "$20"])
ax.set_title("Growth of $1, net of costs — log scale")
ax.legend(facecolor=tl.PANEL, edgecolor=tl.LINE, labelcolor=tl.TEXT, fontsize=9, loc="upper left")
tl.save_fig(fig, "03_nav")

# Chart 2: Monthly turnover
fig, ax = tl.new_fig(9, 3.0)
ax.bar(chart_dates, lo["turnover"], width=20, color=tl.ACCENT, alpha=0.8)
ax.axhline(lo["avg_turnover"], color=tl.WARN, linewidth=1.2, linestyle="--",
           label=f"average {lo['avg_turnover']:.2f}")
ax.set_title("Monthly turnover — fraction of portfolio rebalanced")
ax.set_ylabel("turnover")
ax.legend(facecolor=tl.PANEL, edgecolor=tl.LINE, labelcolor=tl.TEXT, fontsize=9)
tl.save_fig(fig, "03_turnover")

# Chart 3: Return Scatter vs Benchmark
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
fig.patch.set_facecolor(tl.PANEL)
x = np.array(spy) * 100
for ax, (series, label, color) in zip(axes, [
        (lo["net"], "Long-only", tl.ACCENT), (ls0["net"], "Long-short", tl.GOOD)]):
    tl.style_axes(ax)
    y = np.array(series) * 100
    ax.scatter(x, y, s=16, color=color, alpha=0.65, edgecolors="none")
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, slope * xs + intercept, color=tl.TEXT, linewidth=1.3)
    ax.axhline(0, color=tl.MUTED, linewidth=0.8)
    ax.axvline(0, color=tl.MUTED, linewidth=0.8)
    ax.set_title(f"{label}  —  beta = {slope:.2f}")
    ax.set_xlabel("SPY monthly return (%)")
axes[0].set_ylabel("Strategy monthly return (%)")
fig.suptitle("Cross-sectional signal comparison: slope denotes market exposure",
             color=tl.TEXT, fontsize=11)
tl.save_fig(fig, "03_scatter")

# ===========================================================================
# CHECKPOINT
# ===========================================================================
print("\n=== CHECKPOINT ===")
assert len(spy) == 125, f"Expected 125 backtest months, got {len(spy)}"
assert abs(lo_m["sharpe"] - 1.23) < 0.05, f"Long-only Sharpe {lo_m['sharpe']:.2f}, expected ~1.23"
assert abs(lo_m["cagr"] - 0.2664) < 0.01, f"Long-only CAGR {lo_m['cagr']:.4f}, expected ~0.2664"
assert abs(lo_m["max_drawdown"] - (-0.217)) < 0.01, f"Long-only maxDD {lo_m['max_drawdown']:.3f}"
assert abs(lo["avg_turnover"] - 0.55) < 0.03, f"Turnover {lo['avg_turnover']:.2f}, expected ~0.55"
assert abs(lo_beta - 1.10) < 0.05, f"Long-only beta {lo_beta:.2f}, expected ~1.10"
assert abs(ls_beta) < 0.20, f"Long-short beta {ls_beta:.2f} — expected near zero"
assert abs(ls0_m["sharpe"] - 0.29) < 0.05, f"Long-short Sharpe {ls0_m['sharpe']:.2f}, expected ~0.29"
assert abs(portfolio_ir - 0.78) < 0.05, f"Portfolio IR {portfolio_ir:.2f}, expected ~0.78"
assert abs(sum(w_ls.values())) < 1e-9, "Long-short weights must sum to zero"
print(f"OK: Long-only Sharpe {lo_m['sharpe']:.2f} beta {lo_beta:.2f} | "
      f"Long-short Sharpe {ls0_m['sharpe']:.2f} beta {ls_beta:.2f} | IR {portfolio_ir:.2f}")
