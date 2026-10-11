"""Acceptance and regression test suite for quantitative research pipeline.

Verifies end-to-end data integrity, performance metrics, signal ICs, factor models,
web dashboard artifacts, and security policies against benchmark values.

Usage:
    .venv/bin/python steps/check.py
"""

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "lib")]
os.chdir(ROOT)

import numpy as np
import pandas as pd

import analysis
import config
import engine
import timlib as tl
from strategies import BY_ID

passed, failed = [], []


def check(label, actual, expected, tol):
    """Assert value within numerical tolerance without halting execution."""
    ok = actual is not None and abs(actual - expected) <= tol
    (passed if ok else failed).append(label)
    mark = "OK  " if ok else "FAIL"
    got = "None" if actual is None else f"{actual:.4f}"
    print(f"  {mark}  {label:44} {got:>10}   expected {expected:.4f} +/- {tol}")


def check_true(label, condition, detail=""):
    """Assert boolean condition without halting execution."""
    (passed if condition else failed).append(label)
    print(f"  {'OK  ' if condition else 'FAIL'}  {label:44} {detail}")


print("=" * 78)
print(" Acceptance & Regression Test Suite")
print("=" * 78)

# ---------------------------------------------------------------------------
print("\n[1] Environment & Toolchain")
import matplotlib, duckdb
print(f"  python {sys.version.split()[0]}  pandas {pd.__version__}  numpy {np.__version__}  "
      f"matplotlib {matplotlib.__version__}  duckdb {duckdb.__version__}")
check_true("python is 3.12 or newer", sys.version_info >= (3, 12), sys.version.split()[0])

# ---------------------------------------------------------------------------
print("\n[2] Ingestion & Preprocessing")
daily = tl.load_prices()
px = tl.to_monthly(daily)
vol = tl.to_monthly(tl.load_prices(column="Volume"), how="mean")
dates = list(px.index)

check_true("daily prices shape is (2895, 109)", daily.shape == (2895, 109), str(daily.shape))
check_true("index is DatetimeIndex", isinstance(daily.index, pd.DatetimeIndex))
check_true("138 monthly intervals after truncation", len(dates) == 138, f"{len(dates)} months")
check_true("terminal date is 2026-06-30", str(dates[-1].date()) == "2026-06-30", str(dates[-1].date()))
check_true("no non-positive price values", not bool((daily <= 0).any().any()))
check_true("only KHC and PYPL have initial missing observations",
           set(px.columns[px.isna().any()]) == {"KHC", "PYPL"},
           str(sorted(px.columns[px.isna().any()])))

# ---------------------------------------------------------------------------
print("\n[3] Benchmark Performance Metrics")
spy_all = tl.returns(px["SPY"]).dropna()
m = tl.metrics(spy_all)
check("SPY 138mo  $1 grew to", tl.nav(spy_all)[-1], 4.53, 0.05)
check("SPY 138mo  CAGR", m["cagr"], 0.1415, 0.002)
check("SPY 138mo  annual vol", m["vol"], 0.151, 0.005)
check("SPY 138mo  Sharpe (rf=0)", m["sharpe"], 0.96, 0.03)
check("SPY 138mo  max drawdown", m["max_drawdown"], -0.2393, 0.005)

spy = engine.benchmark_returns(px, dates)
sm = tl.metrics(spy)
check_true("backtest window is 125 months", len(spy) == 125, f"{len(spy)} months")
check("SPY 125mo  CAGR", sm["cagr"], 0.1571, 0.002)
check("SPY 125mo  Sharpe", sm["sharpe"], 1.05, 0.03)

# ---------------------------------------------------------------------------
print("\n[4] Long-Only Strategy Execution")
mom = BY_ID["MOM12_1"]
lo = engine.backtest(px, vol, dates, mom["score"], mode="long_only")
lom = tl.metrics(lo["net"])
check("long-only  $1 grew to", tl.nav(lo["net"])[-1], 11.71, 0.35)
check("long-only  CAGR", lom["cagr"], 0.2664, 0.01)
check("long-only  annual vol", lom["vol"], 0.211, 0.008)
check("long-only  Sharpe", lom["sharpe"], 1.23, 0.05)
check("long-only  max drawdown", lom["max_drawdown"], -0.2166, 0.01)
check("long-only  average turnover", lo["avg_turnover"], 0.55, 0.03)
check("long-only  beta to SPY", tl.beta_of(lo["net"], spy), 1.10, 0.05)

# ---------------------------------------------------------------------------
print("\n[5] Dollar-Neutral Strategy Execution")
ls = engine.backtest(px, vol, dates, mom["score"], mode="long_short", borrow_bps=0)
lsm = tl.metrics(ls["net"])
check("long-short  CAGR (0 bps borrow)", lsm["cagr"], 0.0298, 0.006)
check("long-short  Sharpe (0 bps borrow)", lsm["sharpe"], 0.29, 0.05)
check("long-short  annual vol", lsm["vol"], 0.127, 0.008)
check("long-short  beta to SPY", tl.beta_of(ls["net"], spy), -0.07, 0.10)
ls100 = tl.metrics(engine.backtest(px, vol, dates, mom["score"],
                                   mode="long_short", borrow_bps=100)["net"])
check("long-short  Sharpe (100 bps borrow)", ls100["sharpe"], 0.26, 0.05)

w = engine.pick_weights(mom["score"](px, vol, 60), "long_short")
check_true("long-short weights sum to zero", abs(sum(w.values())) < 1e-9,
           f"sum = {sum(w.values()):.2e}")
check_true("long-only weights sum to one",
           abs(sum(engine.pick_weights(mom["score"](px, vol, 60), "long_only").values()) - 1) < 1e-9)

# ---------------------------------------------------------------------------
print("\n[6] Portfolio Information Ratio")
active = np.array(lo["net"]) - np.array(spy)
te = float(active.std(ddof=1) * math.sqrt(12))
pir = float(active.mean() * 12 / te)
check("active return per year", float(active.mean() * 12), 0.1015, 0.01)
check("tracking error", te, 0.1305, 0.01)
check("PORTFOLIO information ratio", pir, 0.78, 0.05)
check("its t-statistic", pir * math.sqrt(len(active) / 12), 2.51, 0.2)

# ---------------------------------------------------------------------------
print("\n[7] Information Coefficient (IC)")
ic_rank = analysis.ic_series(px, vol, dates, mom["score"], "rank")
s = analysis.ic_summary(ic_rank)
check_true("125 monthly ICs", s["months"] == 125, f"{s['months']}")
check("MOM12_1  mean rank IC", s["mean"], 0.0078, 0.004)
check("MOM12_1  IC std", s["std"], 0.227, 0.02)
check("MOM12_1  SIGNAL information ratio", s["ir"], 0.035, 0.02)
check("MOM12_1  t-statistic", s["t"], 0.39, 0.2)
check_true("MOM12_1 IC is not statistically significant", abs(s["t"]) < 2.0, f"t = {s['t']:+.2f}")
check_true("|IC| < 0.05 bounds check", abs(s["mean"]) < 0.05, f"IC = {s['mean']:+.4f}")

pearson = analysis.ic_summary(analysis.ic_series(px, vol, dates, mom["score"], "pearson"))
check("MOM12_1  mean Pearson IC", pearson["mean"], 0.0430, 0.006)
check_true("Pearson IC exceeds rank IC",
           pearson["mean"] / s["mean"] > 3, f"{pearson['mean'] / s['mean']:.1f}x")

hv = analysis.ic_summary(analysis.ic_series(px, vol, dates, BY_ID["HIVOL6"]["score"], "rank"))
check("HIVOL6   mean rank IC", hv["mean"], 0.0407, 0.006)
check("HIVOL6   t-statistic", hv["t"], 2.23, 0.25)
check_true("HIVOL6 clears significance threshold", abs(hv["t"]) > 2.0, f"t = {hv['t']:+.2f}")

# ---------------------------------------------------------------------------
print("\n[8] Quantile Monotonicity")
buckets = analysis.quintile_buckets(px, vol, dates, mom["score"])
means = [float(np.mean(b)) * 100 for b in buckets]
for i, exp in enumerate([1.767, 1.174, 1.299, 1.199, 1.391]):
    check(f"quintile Q{i+1} mean %/month", means[i], exp, 0.15)
check_true("quintiles demonstrate non-monotonic ranking",
           not all(means[i] >= means[i + 1] for i in range(4)),
           "Q5 exceeds Q2/Q3/Q4 returns")

# ---------------------------------------------------------------------------
print("\n[9] Multi-Factor Risk Decomposition")
monthly_returns = px.pct_change(fill_method=None)
FACTORS = analysis.build_factors(monthly_returns, dates)
fit_lo = analysis.ols(lo["net"], FACTORS)
fit_ls = analysis.ols(ls["net"], FACTORS)
check("long-only   R^2", fit_lo["r2"], 0.650, 0.02)
check("long-only   alpha per year", float(fit_lo["beta"][0] * 12), 0.0845, 0.015)
check("long-only   MKT beta", float(fit_lo["beta"][1]), 1.01, 0.06)
check("long-short  R^2", fit_ls["r2"], 0.104, 0.03)
check("long-short  MKT beta", float(fit_ls["beta"][1]), -0.08, 0.10)
check("long-short  GROWTH beta", float(fit_ls["beta"][2]), 0.43, 0.10)
check_true("long-short alpha is not statistically significant", abs(fit_ls["t"][0]) < 1.0,
           f"t = {fit_ls['t'][0]:+.2f}")
check_true("long-short GROWTH loading is statistically significant", fit_ls["t"][2] > 2.0,
           f"t = {fit_ls['t'][2]:+.2f}")

raw = {k: analysis.factor_returns(monthly_returns, dates, k) for k in ("SPY", "QQQ", "IWM")}
spread3 = {"MKT": analysis.factor_returns(monthly_returns, dates, "SPY"),
           "GROWTH": analysis.factor_returns(monthly_returns, dates, "QQQ", "SPY"),
           "SIZE": analysis.factor_returns(monthly_returns, dates, "IWM", "SPY")}
f_raw, f_spread = analysis.ols(lo["net"], raw), analysis.ols(lo["net"], spread3)
check_true("raw and spread models produce identical R^2",
           abs(f_raw["r2"] - f_spread["r2"]) < 1e-9, f"{f_raw['r2']:.6f} vs {f_spread['r2']:.6f}")
check_true("raw and spread models produce identical alpha",
           abs(f_raw["beta"][0] - f_spread["beta"][0]) < 1e-9)
check("raw market coefficient", float(f_raw["beta"][1]), 0.588, 0.03)
check("spread market coefficient", float(f_spread["beta"][1]), 1.041, 0.03)
check("standard error ratio raw/spread", float(f_raw["se"][1] / f_spread["se"][1]), 3.2, 0.4)

v_raw, v_six = analysis.vif(raw), analysis.vif(FACTORS)
check("VIF of raw SPY", v_raw["SPY"], 12.2, 1.0)
check_true("all 6 factors maintain VIF < 2.0",
           max(v_six.values()) < 2.0, f"max = {max(v_six.values()):.2f}")

# ---------------------------------------------------------------------------
print("\n[10] Dashboard Artifacts")
for name in ("overview", "alpha", "factors"):
    p = os.path.join(config.WEB_DATA_FOLDER, name + ".json")
    check_true(f"web/data/{name}.json exists and parses", os.path.exists(p) and
               bool(__import__("json").load(open(p))),
               "" if os.path.exists(p) else "missing")
charts = [f for f in os.listdir(config.CHARTS_FOLDER)
          if f.endswith(".png")] if os.path.exists(config.CHARTS_FOLDER) else []
big = [f for f in charts if os.path.getsize(os.path.join(config.CHARTS_FOLDER, f)) > 10_000]
check_true("at least 10 valid chart artifacts (>10KB)", len(big) >= 10, f"{len(big)} charts")

# ---------------------------------------------------------------------------
print("\n[11] Security & Secret Hygiene")
pems = [os.path.join(r, f) for r, _, fs in os.walk(".") for f in fs
        if f.endswith((".pem", ".key")) and ".venv" not in r]
check_true("no private keys or certificates in repository", not pems, str(pems) if pems else "clean")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
if failed:
    print(f" {len(failed)} FAILED, {len(passed)} passed")
    for f in failed:
        print(f"   FAILED: {f}")
    print("=" * 78)
    sys.exit(1)
print(f" ALL {len(passed)} ACCEPTANCE TESTS PASSED")
print("=" * 78)
