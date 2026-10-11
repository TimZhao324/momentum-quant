"""Cross-engine validation: Compare Pandas vs DuckDB implementations.

Validates that computations across both execution engines (CAGR, total returns,
annualized volatility, sector breakdowns, and cross-sectional momentum ranking)
yield identical numerical results within machine precision tolerances.
"""

import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "lib")]
os.chdir(ROOT)

import duckdb
import numpy as np
import pandas as pd

import config
import timlib as tl

if not os.path.exists(config.DUCKDB_FILE):
    sys.exit(f"{config.DUCKDB_FILE} not found. Run steps/06_database_load.py first.")

con = duckdb.connect(config.DUCKDB_FILE, read_only=True)

px = tl.to_monthly(tl.load_prices())
rets = tl.returns(px)

checks = []


def compare(label, pandas_value, sql_value, tolerance=1e-6):
    """Assert numerical equivalence between Pandas and SQL calculations."""
    ok = abs(pandas_value - sql_value) < tolerance
    checks.append(ok)
    print(f"   Pandas : {pandas_value:>14.8f}")
    print(f"   SQL    : {sql_value:>14.8f}")
    print(f"   -> {'AGREE' if ok else 'DISAGREE!'} (abs diff: {abs(pandas_value - sql_value):.2e})")
    assert ok, f"{label}: pandas {pandas_value} != sql {sql_value}"


# ===========================================================================
# Q1: SPY CAGR over full sample window
# ===========================================================================
print("=== Q1. SPY CAGR: Pandas vs SQL ===")
spy = rets["SPY"].dropna()
n = len(spy)
pandas_cagr = (tl.nav(spy)[-1]) ** (12 / n) - 1

sql_cagr = con.sql("""
    WITH spy AS (
        SELECT close, month_end FROM monthly WHERE ticker = 'SPY' ORDER BY month_end
    ), ends AS (
        SELECT first(close ORDER BY month_end) AS first_close,
               last(close  ORDER BY month_end) AS last_close,
               count(*) - 1                    AS months
        FROM spy
    )
    SELECT pow(last_close / first_close, 12.0 / months) - 1 FROM ends
""").fetchone()[0]
compare("Q1", pandas_cagr, sql_cagr)

# ===========================================================================
# Q2: Ticker with highest total return
# ===========================================================================
print("\n=== Q2. Highest Total Return Asset: Pandas vs SQL ===")
total = (px.iloc[-1] / px.iloc[0] - 1).dropna()
best_ticker = total.idxmax()
pandas_best = float(total.max())

sql_best_row = con.sql("""
    SELECT ticker,
           last(close ORDER BY month_end) / first(close ORDER BY month_end) - 1 AS total_return
    FROM monthly
    GROUP BY ticker
    ORDER BY total_return DESC
    LIMIT 1
""").fetchone()
print(f"   Top performer: Pandas = {best_ticker}, SQL = {sql_best_row[0]}")
assert best_ticker == sql_best_row[0]
compare("Q2", pandas_best, sql_best_row[1])

print("\n   Top 5 performers (DuckDB):")
print(con.sql("""
    SELECT ticker,
           round((last(close ORDER BY month_end) / first(close ORDER BY month_end) - 1) * 100, 0)
               AS total_return_pct
    FROM monthly GROUP BY ticker ORDER BY total_return_pct DESC LIMIT 5
""").df().to_string(index=False))

# ===========================================================================
# Q3: Annualized volatility (AAPL sample check)
# ===========================================================================
print("\n=== Q3. AAPL Annualized Volatility: Pandas vs SQL ===")
pandas_vol = float(rets["AAPL"].std() * math.sqrt(12))

sql_vol = con.sql("""
    SELECT stddev_samp(ret) * sqrt(12)
    FROM monthly_returns
    WHERE ticker = 'AAPL' AND ret IS NOT NULL
""").fetchone()[0]
compare("Q3", pandas_vol, sql_vol)

# ===========================================================================
# Q4: Average monthly return grouped by GICS Sector
# ===========================================================================
print("\n=== Q4. Sector Average Monthly Returns: Pandas vs SQL ===")
sector_of = pd.Series(config.SECTORS)
pandas_sector = (rets[[c for c in rets.columns if c in config.SECTORS]]
                 .mean()
                 .groupby(sector_of)
                 .mean()
                 .sort_values(ascending=False))

sql_sector = con.sql("""
    WITH per_stock AS (
        SELECT ticker, avg(ret) AS avg_ret
        FROM monthly_returns WHERE ret IS NOT NULL GROUP BY ticker
    )
    SELECT s.sector, avg(p.avg_ret) AS avg_ret
    FROM per_stock p JOIN sectors s USING (ticker)
    GROUP BY s.sector ORDER BY avg_ret DESC
""").df()

print(f"   {'Sector':24}{'Pandas':>12}{'SQL':>12}")
for _, row in sql_sector.iterrows():
    p = pandas_sector[row["sector"]]
    print(f"   {row['sector']:24}{p:>12.8f}{row['avg_ret']:>12.8f}")
    assert abs(p - row["avg_ret"]) < 1e-6, f"Sector {row['sector']} mismatch"
checks.append(True)
print("   -> All 11 sectors numerically equivalent")

# ===========================================================================
# Q5: Cross-sectional Momentum 12-1 Signal generation
# ===========================================================================
print("\n=== Q5. Momentum 12-1 Signal Calculation: Pandas vs SQL ===")
target_month = "2020-01-31"
i = list(px.index.strftime("%Y-%m-%d")).index(target_month)
pandas_scores = (px.iloc[i - 1] / px.iloc[i - 12] - 1).dropna().sort_values(ascending=False)

sql_scores = con.sql(f"""
    WITH lagged AS (
        SELECT ticker, month_end,
               lag(close, 1)  OVER w AS px_1,
               lag(close, 12) OVER w AS px_12
        FROM monthly
        WINDOW w AS (PARTITION BY ticker ORDER BY month_end)
    )
    SELECT ticker, px_1 / px_12 - 1 AS score
    FROM lagged
    WHERE month_end = DATE '{target_month}' AND px_1 IS NOT NULL AND px_12 IS NOT NULL
    ORDER BY score DESC
""").df().set_index("ticker")["score"]

print(f"   Month {target_month}: Pandas scored {len(pandas_scores)}, SQL scored {len(sql_scores)}")
print(f"   {'Ticker':8}{'Pandas':>12}{'SQL':>12}")
for t in pandas_scores.head(5).index:
    print(f"   {t:8}{pandas_scores[t]:>12.8f}{sql_scores[t]:>12.8f}")
biggest = float((pandas_scores - sql_scores).abs().max())
print(f"   -> Max discrepancy across all {len(pandas_scores)} assets: {biggest:.2e}")
assert biggest < 1e-9, "SQL signal diverges from Pandas signal"
assert list(pandas_scores.head(10).index) == list(sql_scores.head(10).index), "Top-10 ranking mismatch"
checks.append(True)
print("   -> Identical top-10 constituents selected")

# ===========================================================================
# Benchmark throughput
# ===========================================================================
print("\n=== Execution Latency Comparison ===")
t0 = time.time()
tl.load_prices()
csv_seconds = time.time() - t0

t0 = time.time()
con.sql("SELECT ticker, month_end, close FROM monthly").df()
db_seconds = time.time() - t0
print(f"   CSV file scan into Pandas : {csv_seconds:5.2f} s")
print(f"   DuckDB query to DataFrame : {db_seconds:5.2f} s ({csv_seconds/db_seconds:.0f}x throughput increase)")

con.close()

# ===========================================================================
# CHECKPOINT
# ===========================================================================
print("\n=== CHECKPOINT ===")
assert all(checks), "Discrepancy detected between Pandas and DuckDB calculations"
print(f"OK: All {len(checks)} cross-engine comparisons verified within tolerance.")
