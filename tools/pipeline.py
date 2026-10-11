"""Execution pipeline: Run quantitative analytics and compile web dashboard artifacts.

Computes multi-strategy backtests, cross-sectional IC metrics, multi-factor regressions,
generates publication-ready chart figures, and serializes analytics payloads to JSON.

Usage:
    .venv/bin/python tools/pipeline.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "lib")]
os.chdir(ROOT)

import datetime
import json
import math
import subprocess

import matplotlib.pyplot as plt
import numpy as np

import analysis
import config
import engine
import timlib as tl
from strategies import STRATEGIES, BY_ID

HEADLINE = "MOM12_1"


def clean(obj):
    """Recursively sanitize data structures for standard JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not math.isfinite(v) else round(v, 6)
    return obj


def write(name, doc):
    """Write sanitized analytics document to web/data/<name>.json."""
    os.makedirs(config.WEB_DATA_FOLDER, exist_ok=True)
    path = os.path.join(config.WEB_DATA_FOLDER, name + ".json")
    with open(path, "w") as f:
        json.dump(clean(doc), f, indent=1, allow_nan=False)
    print(f"   {path}  ({os.path.getsize(path)/1024:.1f} KB)")


# ===========================================================================
# 1) Ingest data
# ===========================================================================
print("=== Loading Data ===")
daily = tl.load_prices()
px = tl.to_monthly(daily)
vol = tl.to_monthly(tl.load_prices(column="Volume"), how="mean")
dates = list(px.index)
monthly_returns = px.pct_change(fill_method=None)
spy = engine.benchmark_returns(px, dates)
print(f"{len(dates)} months, {px.shape[1]} tickers, backtest window {len(spy)} months")

# ===========================================================================
# 2) Strategy backtests across long-only and dollar-neutral implementations
# ===========================================================================
print("\n=== Executing Backtests ===")
runs = {}
for s in STRATEGIES:
    lo = engine.backtest(px, vol, dates, s["score"], mode="long_only")
    ls = engine.backtest(px, vol, dates, s["score"], mode="long_short", borrow_bps=0)
    runs[s["id"]] = {"lo": lo, "ls": ls,
                     "lo_m": tl.metrics(lo["net"]), "ls_m": tl.metrics(ls["net"]),
                     "lo_beta": tl.beta_of(lo["net"], spy), "ls_beta": tl.beta_of(ls["net"], spy)}
    print(f"   {s['id']:10} LO Sharpe {runs[s['id']]['lo_m']['sharpe']:5.2f}   "
          f"LS Sharpe {runs[s['id']]['ls_m']['sharpe']:5.2f}")

head = runs[HEADLINE]
lo, ls = head["lo"], head["ls"]
spy_m = tl.metrics(spy)

active = np.array(lo["net"]) - np.array(spy)
tracking_error = float(active.std(ddof=1) * math.sqrt(12))
portfolio_ir = float(active.mean() * 12 / tracking_error)
ir_t = portfolio_ir * math.sqrt(len(active) / 12)

# ===========================================================================
# 3) Information coefficients & predictive ranking diagnostics
# ===========================================================================
print("\n=== Calculating Information Coefficients ===")
ics = {}
for s in STRATEGIES:
    series = analysis.ic_series(px, vol, dates, s["score"], "rank")
    ics[s["id"]] = {"series": series, **analysis.ic_summary(series)}
    print(f"   {s['id']:10} IC {ics[s['id']]['mean']:+.4f}  t {ics[s['id']]['t']:+.2f}")

ic_pearson = analysis.ic_summary(analysis.ic_series(px, vol, dates, BY_ID[HEADLINE]["score"], "pearson"))
buckets = analysis.quintile_buckets(px, vol, dates, BY_ID[HEADLINE]["score"])
bucket_means = [float(np.mean(b)) for b in buckets]
spread_series = [a - b for a, b in zip(buckets[0], buckets[-1])]
spread_mean = float(np.mean(spread_series))
spread_t = spread_mean / (float(np.std(spread_series, ddof=1)) / math.sqrt(len(spread_series)))

# ===========================================================================
# 4) Multi-factor risk decomposition
# ===========================================================================
print("\n=== Fitting Factor Models ===")
FACTORS = analysis.build_factors(monthly_returns, dates)
fit_lo = analysis.ols(lo["net"], FACTORS)
fit_ls = analysis.ols(ls["net"], FACTORS)
raw = {k: analysis.factor_returns(monthly_returns, dates, k) for k in ("SPY", "QQQ", "IWM")}
spread3 = {"MKT": analysis.factor_returns(monthly_returns, dates, "SPY"),
           "GROWTH": analysis.factor_returns(monthly_returns, dates, "QQQ", "SPY"),
           "SIZE": analysis.factor_returns(monthly_returns, dates, "IWM", "SPY")}
fit_raw, fit_spread = analysis.ols(lo["net"], raw), analysis.ols(lo["net"], spread3)
print(f"   long-only  R2 {fit_lo['r2']:.3f}  alpha {fit_lo['beta'][0]*12:+.2%} (t={fit_lo['t'][0]:.2f})")
print(f"   long-short R2 {fit_ls['r2']:.3f}  alpha {fit_ls['beta'][0]*12:+.2%} (t={fit_ls['t'][0]:.2f})")


def fit_json(fit):
    """Format OLS regression result as dictionary."""
    return {"names": fit["names"],
            "beta": [float(b) for b in fit["beta"]],
            "se": [float(b) for b in fit["se"]],
            "t": [float(b) for b in fit["t"]],
            "p": [float(b) for b in fit["p"]],
            "r2": fit["r2"], "adj_r2": fit["adj_r2"],
            "alpha_annual": float(fit["beta"][0] * 12),
            "alpha_se_annual": float(fit["se"][0] * 12),
            "n": fit["n"], "k": fit["k"]}


# ===========================================================================
# 5) Generate figures
# ===========================================================================
print("\n=== Generating Charts ===")
chart_dates = lo["dates"]
nav_lo, nav_ls, nav_spy = tl.nav(lo["net"])[1:], tl.nav(ls["net"])[1:], tl.nav(spy)[1:]

# 03_nav
fig, ax = tl.new_fig(9, 4.6)
ax.plot(chart_dates, nav_lo, color=tl.ACCENT, linewidth=1.7, label="Momentum 12-1, long-only")
ax.plot(chart_dates, nav_ls, color=tl.GOOD, linewidth=1.4, label="Momentum 12-1, long-short")
ax.plot(chart_dates, nav_spy, color=tl.GRAY, linewidth=1.8, label="SPY (benchmark)")
ax.set_yscale("log")
ax.set_yticks([0.5, 1, 2, 5, 10, 20])
ax.set_yticklabels(["$0.5", "$1", "$2", "$5", "$10", "$20"])
ax.set_title("Growth of $1, net of trading costs — log scale")
ax.legend(facecolor=tl.PANEL, edgecolor=tl.LINE, labelcolor=tl.TEXT, fontsize=9, loc="upper left")
tl.save_fig(fig, "03_nav")

# 03_drawdown
peak, dd = nav_lo[0], []
for v in nav_lo:
    peak = max(peak, v)
    dd.append((v / peak - 1) * 100)
fig, ax = tl.new_fig(9, 3.0)
ax.fill_between(chart_dates, dd, 0, color=tl.BAD, alpha=0.35)
ax.plot(chart_dates, dd, color=tl.BAD, linewidth=1.1)
ax.set_title(f"Long-only drawdown (max: {min(dd):.1f}%)")
ax.set_ylabel("% below peak")
tl.save_fig(fig, "03_drawdown")

# 03_turnover
fig, ax = tl.new_fig(9, 3.0)
ax.bar(chart_dates, lo["turnover"], width=20, color=tl.ACCENT, alpha=0.8)
ax.axhline(lo["avg_turnover"], color=tl.WARN, linewidth=1.2, linestyle="--",
           label=f"average {lo['avg_turnover']:.2f}")
ax.set_title("Monthly turnover")
ax.legend(facecolor=tl.PANEL, edgecolor=tl.LINE, labelcolor=tl.TEXT, fontsize=9)
tl.save_fig(fig, "03_turnover")

# 03_scatter
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
fig.patch.set_facecolor(tl.PANEL)
x = np.array(spy) * 100
for ax, (series, label, color) in zip(axes, [(lo["net"], "Long-only", tl.ACCENT),
                                             (ls["net"], "Long-short", tl.GOOD)]):
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
fig.suptitle("Strategy vs Benchmark monthly returns: slope represents beta",
             color=tl.TEXT, fontsize=11)
tl.save_fig(fig, "03_scatter")

# 04_ic_series
ic_dates = chart_dates[:ics[HEADLINE]["months"]]
fig, ax = tl.new_fig(9, 3.6)
ax.bar(ic_dates, ics[HEADLINE]["series"], width=20,
       color=[tl.GOOD if v > 0 else tl.BAD for v in ics[HEADLINE]["series"]], alpha=0.85)
ax.axhline(ics[HEADLINE]["mean"], color=tl.ACCENT, linewidth=1.5,
           label=f"mean {ics[HEADLINE]['mean']:+.4f} (t = {ics[HEADLINE]['t']:+.2f})")
ax.axhline(0, color=tl.MUTED, linewidth=0.8)
ax.set_title("Monthly rank IC — Momentum 12-1")
ax.legend(facecolor=tl.PANEL, edgecolor=tl.LINE, labelcolor=tl.TEXT, fontsize=9)
tl.save_fig(fig, "04_ic_series")

# 04_ic_cumulative
fig, ax = tl.new_fig(9, 3.6)
for sid in ("HIVOL6", "MOM12_1", "REV1"):
    ax.plot(chart_dates[:ics[sid]["months"]], np.cumsum(ics[sid]["series"]),
            linewidth=1.5, label=f"{sid} (t = {ics[sid]['t']:+.2f})")
ax.axhline(0, color=tl.MUTED, linewidth=0.8)
ax.set_title("Cumulative rank IC trajectories")
ax.legend(facecolor=tl.PANEL, edgecolor=tl.LINE, labelcolor=tl.TEXT, fontsize=9)
tl.save_fig(fig, "04_ic_cumulative")

# 04_quintiles
fig, ax = tl.new_fig(7.5, 3.8)
vals = [m * 100 for m in bucket_means]
bars = ax.bar([f"Q{i+1}" for i in range(5)], vals,
              color=[tl.GOOD] + [tl.ACCENT] * 3 + [tl.BAD], alpha=0.9)
for b, m in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, m + 0.03, f"{m:.2f}", ha="center",
            color=tl.TEXT, fontsize=9)
ax.set_title("Average next-month return by score quintile — Momentum 12-1")
ax.set_ylabel("% per month")
ax.set_xlabel("Q1 = highest rank, Q5 = lowest rank")
tl.save_fig(fig, "04_quintiles")

# 05_betas
fig, ax = tl.new_fig(9, 4.0)
names = fit_lo["names"][1:]
xpos = np.arange(len(names))
for offset, fit, label, color in [(-0.19, fit_lo, "long-only", tl.ACCENT),
                                  (+0.19, fit_ls, "long-short", tl.GOOD)]:
    ax.bar(xpos + offset, fit["beta"][1:], 0.38, yerr=1.96 * fit["se"][1:],
           color=color, alpha=0.9, label=label,
           error_kw={"ecolor": tl.TEXT, "elinewidth": 1.1, "capsize": 3})
ax.axhline(0, color=tl.MUTED, linewidth=1)
ax.set_xticks(xpos)
ax.set_xticklabels(names)
ax.set_title("Factor exposures with 95% confidence intervals")
ax.set_ylabel("beta")
ax.legend(facecolor=tl.PANEL, edgecolor=tl.LINE, labelcolor=tl.TEXT, fontsize=9)
tl.save_fig(fig, "05_betas")

# 05_fit
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
fig.patch.set_facecolor(tl.PANEL)
for ax, fit, series, label, color in zip(axes, [fit_lo, fit_ls], [lo["net"], ls["net"]],
                                         ["Long-only", "Long-short"], [tl.ACCENT, tl.GOOD]):
    tl.style_axes(ax)
    a, f = np.array(series) * 100, fit["fitted"] * 100
    ax.scatter(f, a, s=16, color=color, alpha=0.65, edgecolors="none")
    lims = [min(f.min(), a.min()), max(f.max(), a.max())]
    ax.plot(lims, lims, color=tl.TEXT, linewidth=1.1, linestyle="--")
    ax.set_title(f"{label}  —  R2 = {fit['r2']:.2f}")
    ax.set_xlabel("predicted by 6 factors (%)")
axes[0].set_ylabel("realized returns (%)")
fig.suptitle("Variance explained by 6-factor model", color=tl.TEXT, fontsize=11)
tl.save_fig(fig, "05_fit")

# 05_residuals
fig, ax = tl.new_fig(9, 3.4)
ax.bar(chart_dates, fit_lo["resid"] * 100, width=20, color=tl.ACCENT, alpha=0.8,
       label=f"long-only (alpha {fit_lo['beta'][0]*12:+.1%}/yr, t = {fit_lo['t'][0]:.2f})")
ax.axhline(0, color=tl.MUTED, linewidth=0.8)
ax.set_title("Regression residuals (unexplained variance)")
ax.set_ylabel("residual (%)")
ax.legend(facecolor=tl.PANEL, edgecolor=tl.LINE, labelcolor=tl.TEXT, fontsize=9)
tl.save_fig(fig, "05_residuals")

# ===========================================================================
# 6) Serialize payload documents to JSON
# ===========================================================================
print("\n=== Serializing JSON Artifacts ===")
today = datetime.date.today().isoformat()
window = {"start": str(chart_dates[0].date()), "end": str(chart_dates[-1].date()),
          "months": len(spy), "universe": px.shape[1]}

write("overview", {
    "generated_at": today, "window": window,
    "headline": {
        "portfolio_ir": portfolio_ir, "ir_t": ir_t,
        "active_return": float(active.mean() * 12), "tracking_error": tracking_error,
        "lo_sharpe": head["lo_m"]["sharpe"], "lo_cagr": head["lo_m"]["cagr"],
        "lo_mdd": head["lo_m"]["max_drawdown"], "lo_beta": head["lo_beta"],
        "lo_nav": nav_lo[-1], "lo_turnover": lo["avg_turnover"],
        "lo_cost_bps": lo["cost_drag_bps"],
        "ls_sharpe": head["ls_m"]["sharpe"], "ls_cagr": head["ls_m"]["cagr"],
        "ls_beta": head["ls_beta"], "ls_nav": nav_ls[-1],
        "spy_sharpe": spy_m["sharpe"], "spy_cagr": spy_m["cagr"], "spy_nav": nav_spy[-1],
        "signal_ic": ics[HEADLINE]["mean"], "signal_t": ics[HEADLINE]["t"],
    },
    "nav_curve": [{"date": str(d.date()), "long_only": a, "long_short": b, "spy": c}
                  for d, a, b, c in zip(chart_dates, nav_lo, nav_ls, nav_spy)],
    "strategies": [{
        "id": s["id"], "name": s["name"], "sleeve": s["sleeve"],
        "lo_sharpe": runs[s["id"]]["lo_m"]["sharpe"], "lo_cagr": runs[s["id"]]["lo_m"]["cagr"],
        "lo_mdd": runs[s["id"]]["lo_m"]["max_drawdown"], "lo_beta": runs[s["id"]]["lo_beta"],
        "ls_sharpe": runs[s["id"]]["ls_m"]["sharpe"], "ls_cagr": runs[s["id"]]["ls_m"]["cagr"],
        "ls_beta": runs[s["id"]]["ls_beta"],
        "turnover": runs[s["id"]]["lo"]["avg_turnover"],
        "cost_bps": runs[s["id"]]["lo"]["cost_drag_bps"],
    } for s in STRATEGIES],
})

write("alpha", {
    "generated_at": today, "window": window,
    "ic_table": [{
        "id": s["id"], "name": s["name"], "sleeve": s["sleeve"],
        "ic_mean": ics[s["id"]]["mean"], "ic_std": ics[s["id"]]["std"],
        "signal_ir": ics[s["id"]]["ir"], "t": ics[s["id"]]["t"], "p": ics[s["id"]]["p"],
        "hit_rate": ics[s["id"]]["hit_rate"],
        "verdict": "significant" if abs(ics[s["id"]]["t"]) > 2 else "noise",
    } for s in STRATEGIES],
    "pearson_vs_rank": {"pearson": ic_pearson["mean"], "rank": ics[HEADLINE]["mean"],
                        "ratio": ic_pearson["mean"] / ics[HEADLINE]["mean"]},
    "ir_names": {"signal_ir": ics[HEADLINE]["ir"], "portfolio_ir": portfolio_ir,
                 "active_return": float(active.mean() * 12), "tracking_error": tracking_error},
    "grinold": {"breadth": px.shape[1] * 12,
                "predicted_ir": ics[HEADLINE]["mean"] * math.sqrt(px.shape[1] * 12),
                "measured_sharpe": head["ls_m"]["sharpe"]},
    "quintiles": {
        "labels": [f"Q{i+1}" for i in range(5)],
        "monthly_pct": [m * 100 for m in bucket_means],
        "monotone": all(bucket_means[i] >= bucket_means[i + 1] for i in range(4)),
        "spread_monthly_pct": spread_mean * 100, "spread_annual_pct": spread_mean * 12 * 100,
        "spread_t": spread_t,
    },
})

write("factors", {
    "generated_at": today, "window": window,
    "long_only": fit_json(fit_lo), "long_short": fit_json(fit_ls),
    "vif": {"raw": analysis.vif(raw), "spread": analysis.vif(spread3),
            "six": analysis.vif(FACTORS)},
    "parameterisation": {
        "raw": {"r2": fit_raw["r2"], "alpha_annual": float(fit_raw["beta"][0] * 12),
                "market_beta": float(fit_raw["beta"][1]), "market_se": float(fit_raw["se"][1])},
        "spread": {"r2": fit_spread["r2"], "alpha_annual": float(fit_spread["beta"][0] * 12),
                   "market_beta": float(fit_spread["beta"][1]), "market_se": float(fit_spread["se"][1])},
        "se_ratio": float(fit_raw["se"][1] / fit_spread["se"][1]),
    },
    "correlations": {a: {b: float(np.corrcoef(raw[a], raw[b])[0, 1]) for b in raw} for a in raw},
})

try:
    server = subprocess.run(["bash", "steps/00_hello_server.sh"], capture_output=True,
                            text=True, timeout=60).stdout
except Exception as e:
    server = f"(could not run steps/00_hello_server.sh: {e})"
write("server", {"generated_at": today, "output": server})

print("\nPipeline execution complete.")
