"""Multi-factor risk attribution and linear regression decomposition.

Decomposes portfolio returns across market, style, macro, and commodity factor proxies
using Ordinary Least Squares (OLS) and Variance Inflation Factor (VIF) collinearity diagnostics.

Outputs:
    web/charts/05_betas.png      Factor beta loadings with 95% confidence intervals
    web/charts/05_fit.png        Model predicted vs realized returns
    web/charts/05_residuals.png  Time series of regression residuals (unexplained variance)
"""

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "lib")]
os.chdir(ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import timlib as tl
import engine
from analysis import ols, vif, build_factors, factor_returns as _factor_returns
from strategies import BY_ID

# ===========================================================================
# 1) Ingest data and construct factor returns
# ===========================================================================
print("=== 1. Data Ingestion ===")
daily = tl.load_prices()
px = tl.to_monthly(daily)
vol = tl.to_monthly(tl.load_prices(column="Volume"), how="mean")
dates = list(px.index)
monthly_returns = px.pct_change(fill_method=None)


def factor_returns(symbol, subtract=None):
    """Compute monthly factor return series."""
    return _factor_returns(monthly_returns, dates, symbol, subtract)


FACTORS = build_factors(monthly_returns, dates)
print(f"{len(FACTORS)} factors, {len(next(iter(FACTORS.values())))} months: {list(FACTORS)}")

mom = BY_ID["MOM12_1"]
lo = engine.backtest(px, vol, dates, mom["score"], mode="long_only")
ls = engine.backtest(px, vol, dates, mom["score"], mode="long_short", borrow_bps=0)
spy = engine.benchmark_returns(px, dates)

# ===========================================================================
# 2) Ordinary Least Squares (OLS) factor attribution
# ===========================================================================
print("\n=== 2. Multi-Factor Regression Decomposition ===")


def show(fit, title):
    """Format and display regression summary table."""
    print(f"\n{title}")
    print(f"{'':10}{'coef':>10}{'std err':>10}{'t':>8}{'p':>8}")
    for i, name in enumerate(fit["names"]):
        star = " *" if abs(fit["t"][i]) > 2 else ""
        if name == "ALPHA":
            print(f"{name:10}{fit['beta'][i]*12:>9.2%}{fit['se'][i]*12:>10.2%}"
                  f"{fit['t'][i]:>8.2f}{fit['p'][i]:>8.3f}{star}   <- per year")
        else:
            print(f"{name:10}{fit['beta'][i]:>10.3f}{fit['se'][i]:>10.3f}"
                  f"{fit['t'][i]:>8.2f}{fit['p'][i]:>8.3f}{star}")
    print(f"R^2 = {fit['r2']:.3f}   adjusted R^2 = {fit['adj_r2']:.3f}   "
          f"n = {fit['n']}, k = {fit['k']}, dof = {fit['n'] - fit['k']}")


fit_lo = ols(lo["net"], FACTORS)
show(fit_lo, "Long-Only momentum book explained by 6 factors:")

fit_ls = ols(ls["net"], FACTORS)
show(fit_ls, "Long-Short momentum book explained by 6 factors:")

# ===========================================================================
# 3) Multicollinearity diagnostics: Raw ETFs vs Orthogonalized Spreads
# ===========================================================================
print("\n=== 3. Multicollinearity Diagnostics (VIF) ===")

raw = {"SPY": factor_returns("SPY"), "QQQ": factor_returns("QQQ"), "IWM": factor_returns("IWM")}
spread = {"MKT": factor_returns("SPY"),
          "GROWTH": factor_returns("QQQ", "SPY"),
          "SIZE": factor_returns("IWM", "SPY")}

print("\nCorrelations among raw ETF returns:")
raw_df = pd.DataFrame(raw)
print(raw_df.corr().round(2).to_string())

print(f"\nVIF, raw ETFs   : " + "  ".join(f"{k} {v:5.1f}" for k, v in vif(raw).items()))
print(f"VIF, spreads    : " + "  ".join(f"{k} {v:5.2f}" for k, v in vif(spread).items()))
print(f"VIF, 6-factor   : " + "  ".join(f"{k} {v:.2f}" for k, v in vif(FACTORS).items()))

fit_raw = ols(lo["net"], raw)
fit_spread = ols(lo["net"], spread)

print(f"\n{'':22}{'R^2':>8}{'alpha/yr':>11}{'market coef':>13}{'std err':>10}{'t':>7}")
print(f"{'raw SPY/QQQ/IWM':22}{fit_raw['r2']:>8.4f}{fit_raw['beta'][0]*12:>11.2%}"
      f"{fit_raw['beta'][1]:>13.3f}{fit_raw['se'][1]:>10.3f}{fit_raw['t'][1]:>7.2f}")
print(f"{'MKT/GROWTH/SIZE':22}{fit_spread['r2']:>8.4f}{fit_spread['beta'][0]*12:>11.2%}"
      f"{fit_spread['beta'][1]:>13.3f}{fit_spread['se'][1]:>10.3f}{fit_spread['t'][1]:>7.2f}")

# ===========================================================================
# 4) Visualizations
# ===========================================================================
print("\n=== 4. Generating Visualizations ===")

# Chart 1: Factor exposures with error bars
fig, ax = tl.new_fig(9, 4.0)
names = fit_lo["names"][1:]
x = np.arange(len(names))
width = 0.38
for offset, fit, label, color in [(-width/2, fit_lo, "long-only", tl.ACCENT),
                                  (+width/2, fit_ls, "long-short", tl.GOOD)]:
    ax.bar(x + offset, fit["beta"][1:], width, yerr=1.96 * fit["se"][1:],
           color=color, alpha=0.9, label=label,
           error_kw={"ecolor": tl.TEXT, "elinewidth": 1.1, "capsize": 3})
ax.axhline(0, color=tl.MUTED, linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_title("Factor exposures with 95% confidence intervals")
ax.set_ylabel("beta")
ax.legend(facecolor=tl.PANEL, edgecolor=tl.LINE, labelcolor=tl.TEXT, fontsize=9)
tl.save_fig(fig, "05_betas")

# Chart 2: Model fitted vs realized returns
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
fig.patch.set_facecolor(tl.PANEL)
for ax, fit, label, color in zip(axes, [fit_lo, fit_ls],
                                 ["Long-only", "Long-short"], [tl.ACCENT, tl.GOOD]):
    tl.style_axes(ax)
    a = np.array(lo["net"] if label == "Long-only" else ls["net"]) * 100
    f = fit["fitted"] * 100
    ax.scatter(f, a, s=16, color=color, alpha=0.65, edgecolors="none")
    lims = [min(f.min(), a.min()), max(f.max(), a.max())]
    ax.plot(lims, lims, color=tl.TEXT, linewidth=1.1, linestyle="--")
    ax.set_title(f"{label}  —  R2 = {fit['r2']:.2f}")
    ax.set_xlabel("predicted by 6 factors (%)")
axes[0].set_ylabel("realized returns (%)")
fig.suptitle("Variance explained by multi-factor model", color=tl.TEXT, fontsize=11)
tl.save_fig(fig, "05_fit")

# Chart 3: Model residuals
fig, ax = tl.new_fig(9, 3.4)
res_dates = lo["dates"]
ax.bar(res_dates, fit_lo["resid"] * 100, width=20, color=tl.ACCENT, alpha=0.8,
       label=f"long-only (alpha {fit_lo['beta'][0]*12:+.1%}/yr, t={fit_lo['t'][0]:.2f})")
ax.axhline(0, color=tl.MUTED, linewidth=0.8)
ax.set_title("Regression Residuals (unexplained variance)")
ax.set_ylabel("residual (%)")
ax.legend(facecolor=tl.PANEL, edgecolor=tl.LINE, labelcolor=tl.TEXT, fontsize=9)
tl.save_fig(fig, "05_residuals")

# ===========================================================================
# CHECKPOINT
# ===========================================================================
print("\n=== CHECKPOINT ===")
assert abs(fit_lo["r2"] - 0.650) < 0.02, f"Long-only R2 {fit_lo['r2']:.3f}, expected ~0.650"
assert abs(fit_lo["beta"][1] - 1.01) < 0.06, f"Long-only MKT beta {fit_lo['beta'][1]:.2f}, expected ~1.01"
assert abs(fit_ls["r2"] - 0.104) < 0.03, f"Long-short R2 {fit_ls['r2']:.3f}, expected ~0.104"
assert abs(fit_ls["t"][0]) < 1.0, "Long-short alpha should not be statistically significant"
assert fit_ls["t"][2] > 2.0, "Long-short GROWTH loading should be significant"
assert abs(fit_raw["r2"] - fit_spread["r2"]) < 1e-9, "Raw and spread fits must yield identical R2"
assert abs(fit_raw["beta"][0] - fit_spread["beta"][0]) < 1e-9, "Alpha must be identical"
assert fit_raw["se"][1] / fit_spread["se"][1] > 2.5, "Raw standard error should be significantly wider"
assert max(vif(raw).values()) > 8, "Raw ETFs should exhibit high collinearity"
assert max(vif(FACTORS).values()) < 2.0, "Orthogonalized factors should show low VIF (< 2.0)"
print(f"OK: Long-only R2 {fit_lo['r2']:.3f} alpha {fit_lo['beta'][0]*12:+.2%} (t={fit_lo['t'][0]:.2f}) | "
      f"Long-short R2 {fit_ls['r2']:.3f} | Raw/Spread SE ratio {fit_raw['se'][1]/fit_spread['se'][1]:.1f}x")
