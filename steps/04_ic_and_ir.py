"""Information Coefficient (IC) and Information Ratio (IR) cross-sectional analysis.

Evaluates cross-sectional ranking predictive power (Pearson vs Spearman Rank IC),
t-statistics, Grinold's fundamental law cross-checks, and quantile bucket monotonicity.

Outputs:
    web/charts/04_ic_series.png     Monthly rank IC trajectory
    web/charts/04_ic_cumulative.png Cumulative IC trajectories across candidate signals
    web/charts/04_quintiles.png     Average forward return by score quintile bucket
"""

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "lib")]
os.chdir(ROOT)

import numpy as np
import pandas as pd

import config
import timlib as tl
import engine
from analysis import ic_series as _ic_series, ic_summary, quintile_buckets
from strategies import STRATEGIES, BY_ID

# ===========================================================================
# 1) Ingest data
# ===========================================================================
print("=== 1. Data Ingestion ===")
daily = tl.load_prices()
px = tl.to_monthly(daily)
vol = tl.to_monthly(tl.load_prices(column="Volume"), how="mean")
dates = list(px.index)
print(f"{len(dates)} months, {px.shape[1]} tickers")

# ===========================================================================
# 2) Single period IC step-by-step verification
# ===========================================================================
print("\n=== 2. Single-Month Cross-Sectional IC Inspection ===")
mom = BY_ID["MOM12_1"]
i = 60

scores = mom["score"](px, vol, i)
forward = px.iloc[i + 1] / px.iloc[i] - 1

both = pd.concat([scores, forward], axis=1).dropna()
both.columns = ["score", "forward_return"]
print(f"Month {dates[i].date()} -> earning {dates[i+1].date()}, {len(both)} tickers available")

pearson = both["score"].corr(both["forward_return"])
spearman = both["score"].rank().corr(both["forward_return"].rank())
print(f"   Pearson  IC = {pearson:+.4f} (parametric)")
print(f"   Spearman IC = {spearman:+.4f} (rank-based)")

# ===========================================================================
# 3) Full sample IC time series
# ===========================================================================
print("\n=== 3. Sample Window IC Metrics ===")

def ic_series(score_fn, method="rank"):
    """Compute monthly cross-sectional IC across full backtest sample."""
    return _ic_series(px, vol, dates, score_fn, method)

ic_rank = ic_series(mom["score"], "rank")
ic_pearson = ic_series(mom["score"], "pearson")

summary = ic_summary(ic_rank)
T = summary["months"]
ic_mean, ic_std = summary["mean"], summary["std"]
signal_ir, t_stat, p_value, hit = summary["ir"], summary["t"], summary["p"], summary["hit_rate"]

print(f"Months            T = {T}")
print(f"Mean Rank IC        = {ic_mean:+.4f}")
print(f"Std  Rank IC        = {ic_std:.4f}")
print(f"SIGNAL IR           = {signal_ir:+.4f} (mean / std)")
print(f"t-statistic         = {t_stat:+.2f} (IR * sqrt(T))")
print(f"p-value             = {p_value:.3f}")
print(f"Hit Rate (IC > 0)   = {hit:.1%}")
print(f"Mean Pearson IC     = {np.mean(ic_pearson):+.4f}")

# ===========================================================================
# 4) Fundamental law of active management cross-check
# ===========================================================================
print("\n=== 4. Grinold Fundamental Law Cross-Check ===")
breadth = px.shape[1] * 12
predicted = ic_mean * math.sqrt(breadth)
ls = engine.backtest(px, vol, dates, mom["score"], mode="long_short", borrow_bps=0)
measured = tl.metrics(ls["net"])["sharpe"]
print(f"Breadth: {px.shape[1]} assets x 12 periods = {breadth} bets/yr")
print(f"Predicted IR = IC * sqrt(breadth) = {ic_mean:.4f} * {math.sqrt(breadth):.1f} = {predicted:.2f}")
print(f"Measured dollar-neutral Sharpe = {measured:.2f}")

# ===========================================================================
# 5) Signal IR vs Portfolio IR
# ===========================================================================
print("\n=== 5. Signal IR vs Portfolio IR ===")
lo = engine.backtest(px, vol, dates, mom["score"], mode="long_only")
spy = engine.benchmark_returns(px, dates)
active = np.array(lo["net"]) - np.array(spy)
portfolio_ir = (active.mean() * 12) / (active.std(ddof=1) * math.sqrt(12))

print(f"  Signal IR    = mean(IC) / std(IC)                     = {signal_ir:+.3f}")
print(f"  Portfolio IR = active return / tracking error (ann.) = {portfolio_ir:+.3f}")

# ===========================================================================
# 6) Quintile bucket monotonicity analysis
# ===========================================================================
print("\n=== 6. Quintile Monotonicity Analysis ===")
buckets = quintile_buckets(px, vol, dates, mom["score"])

means = [float(np.mean(b)) * 100 for b in buckets]
print(f"{'Bucket':10}{'Avg Return (%/mo)':>20}")
for b, m in enumerate(means):
    tag = "(top)" if b == 0 else "(bottom)" if b == 4 else ""
    print(f"Q{b+1} {tag:8}{m:>12.3f}")

spread = [a - b for a, b in zip(buckets[0], buckets[4])]
sp_mean = float(np.mean(spread))
sp_sharpe = sp_mean / float(np.std(spread, ddof=1)) * math.sqrt(12)
sp_t = sp_mean / (float(np.std(spread, ddof=1)) / math.sqrt(len(spread)))
print(f"\nQ1 - Q5 spread: {sp_mean*100:.3f}%/mo ({sp_mean*12*100:.2f}%/yr), "
      f"Sharpe: {sp_sharpe:.2f}, t = {sp_t:.2f}")

monotone = all(means[idx] >= means[idx + 1] for idx in range(4))
print(f"Strict monotonicity preserved: {monotone}")

# ===========================================================================
# 7) Cross-signal comparison
# ===========================================================================
print("\n=== 7. Signal Universe Comparison ===")
print(f"{'id':10}{'sleeve':12}{'rank IC':>10}{'IC std':>9}{'signal IR':>11}{'t':>7}{'verdict':>12}")
ic_table = {}
for s in STRATEGIES:
    ser = ic_series(s["score"], "rank")
    summ = ic_summary(ser)
    m, sd, ir, t = summ["mean"], summ["std"], summ["ir"], summ["t"]
    ic_table[s["id"]] = {"mean": m, "std": sd, "ir": ir, "t": t, "series": ser}
    verdict = "Significant" if abs(t) > 2 else "Noise"
    print(f"{s['id']:10}{s['sleeve']:12}{m:>+10.4f}{sd:>9.3f}{ir:>+11.3f}{t:>+7.2f}{verdict:>12}")

# ===========================================================================
# 8) Visualizations
# ===========================================================================
print("\n=== 8. Generating Visualizations ===")
ic_dates = [dates[idx + 1] for idx in range(config.WARMUP, len(dates) - 1)][:T]

# Chart 1: Monthly Rank IC
fig, ax = tl.new_fig(9, 3.6)
colors = [tl.GOOD if v > 0 else tl.BAD for v in ic_rank]
ax.bar(ic_dates, ic_rank, width=20, color=colors, alpha=0.85)
ax.axhline(ic_mean, color=tl.ACCENT, linewidth=1.5, label=f"mean {ic_mean:+.4f}")
ax.axhline(0, color=tl.MUTED, linewidth=0.8)
ax.set_title("Monthly Rank IC — Momentum 12-1")
ax.set_ylabel("Rank IC")
ax.legend(facecolor=tl.PANEL, edgecolor=tl.LINE, labelcolor=tl.TEXT, fontsize=9)
tl.save_fig(fig, "04_ic_series")

# Chart 2: Cumulative IC
fig, ax = tl.new_fig(9, 3.6)
for sid in ("MOM12_1", "HIVOL6", "REV1"):
    ax.plot(ic_dates[:len(ic_table[sid]["series"])], np.cumsum(ic_table[sid]["series"]),
            linewidth=1.5, label=f"{sid} (t={ic_table[sid]['t']:+.2f})")
ax.axhline(0, color=tl.MUTED, linewidth=0.8)
ax.set_title("Cumulative Rank IC Trajectories")
ax.set_ylabel("Cumulative IC")
ax.legend(facecolor=tl.PANEL, edgecolor=tl.LINE, labelcolor=tl.TEXT, fontsize=9)
tl.save_fig(fig, "04_ic_cumulative")

# Chart 3: Quintile Return Bars
fig, ax = tl.new_fig(7.5, 3.8)
bars = ax.bar([f"Q{idx+1}" for idx in range(5)], means,
              color=[tl.GOOD] + [tl.ACCENT] * 3 + [tl.BAD], alpha=0.9)
ax.set_title("Average Next-Month Return by Score Quintile — Momentum 12-1")
ax.set_ylabel("% per month")
ax.set_xlabel("Q1 = highest rank, Q5 = lowest rank")
for b, m in zip(bars, means):
    ax.text(b.get_x() + b.get_width() / 2, m + 0.03, f"{m:.2f}", ha="center",
            color=tl.TEXT, fontsize=9)
tl.save_fig(fig, "04_quintiles")

# ===========================================================================
# CHECKPOINT
# ===========================================================================
print("\n=== CHECKPOINT ===")
assert T == 125, f"Expected 125 monthly ICs, got {T}"
assert abs(ic_mean) < 0.05, f"|mean IC| = {ic_mean:.4f} is too large"
assert abs(ic_mean - 0.0078) < 0.004, f"Rank IC {ic_mean:.4f}, expected ~0.0078"
assert abs(np.mean(ic_pearson) - 0.0430) < 0.006, f"Pearson IC {np.mean(ic_pearson):.4f}, expected ~0.043"
assert abs(t_stat) < 2.0, "Momentum IC t-stat should not clear statistical significance"
assert abs(ic_table["HIVOL6"]["t"] - 2.23) < 0.25, f"HIVOL6 t {ic_table['HIVOL6']['t']:.2f}, expected ~2.23"
assert not monotone, "Quintiles must show non-monotonic behavior on this universe"
assert abs(portfolio_ir - 0.78) < 0.05, f"Portfolio IR {portfolio_ir:.2f}, expected ~0.78"
print(f"OK: Rank IC {ic_mean:+.4f} (t={t_stat:+.2f}) | Pearson {np.mean(ic_pearson):+.4f} | Portfolio IR {portfolio_ir:.2f}")
