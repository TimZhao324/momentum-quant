"""ETL pipeline: Ingest price history CSVs into columnar DuckDB database.

Loads 109 ticker CSV files into DuckDB, builds monthly aggregation tables,
computes period returns using window functions, and verifies data integrity.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "lib")]
os.chdir(ROOT)

import duckdb
import config

DB = config.DUCKDB_FILE

# Rebuild cleanly on each run
if os.path.exists(DB):
    os.remove(DB)
    print(f"Removed previous database file: {DB}")

con = duckdb.connect(DB)
print(f"DuckDB {duckdb.__version__} connected -> {DB}\n")

# ===========================================================================
# 1) Ingest all CSVs into unified 'prices' table
# ===========================================================================
print("=== 1. Ingesting CSV Files ===")

con.sql(f"""
    CREATE TABLE prices AS
    SELECT
        regexp_extract(filename, '([^/]+)\\.csv$', 1) AS ticker,
        CAST("Date" AS DATE)  AS date,
        "Close"               AS close,
        "High"                AS high,
        "Low"                 AS low,
        "Open"                AS open,
        CAST("Volume" AS BIGINT) AS volume
    FROM read_csv('{config.DATA_FOLDER}/*.csv', filename = true, header = true)
    ORDER BY ticker, date
""")

n_rows = con.sql("SELECT count(*) FROM prices").fetchone()[0]
n_tickers = con.sql("SELECT count(DISTINCT ticker) FROM prices").fetchone()[0]
print(f"Prices table: {n_rows:,} rows across {n_tickers} tickers")
print("\nSchema definition:")
print(con.sql("DESCRIBE prices").df().to_string(index=False))
print("\nSample records:")
print(con.sql("SELECT * FROM prices ORDER BY ticker, date LIMIT 5").df().to_string(index=False))

# ===========================================================================
# 2) Data validation checks
# ===========================================================================
print("\n=== 2. Data Integrity Checks ===")

print("\nDate coverage per ticker (minimum history check):")
print(con.sql("""
    SELECT ticker, min(date) AS first_day, max(date) AS last_day, count(*) AS rows
    FROM prices
    GROUP BY ticker
    ORDER BY rows ASC
    LIMIT 4
""").df().to_string(index=False))

print("\nAnomaly inspection:")
print(con.sql("""
    SELECT
        count(*) FILTER (WHERE close <= 0)                  AS non_positive_close,
        count(*) FILTER (WHERE close IS NULL)               AS null_close,
        count(*) FILTER (WHERE high < low)                  AS high_below_low,
        count(*) FILTER (WHERE close > high OR close < low) AS close_outside_range,
        count(*) FILTER (WHERE volume < 0)                  AS negative_volume
    FROM prices
""").df().to_string(index=False))

# ===========================================================================
# 3) Build monthly resampled table
# ===========================================================================
print("\n=== 3. Monthly Resampling ===")

con.sql("""
    CREATE TABLE monthly AS
    SELECT
        ticker,
        last_day(date)              AS month_end,
        arg_max(close, date)        AS close,
        avg(volume)                 AS avg_volume,
        count(*)                    AS trading_days
    FROM prices
    GROUP BY ticker, last_day(date)
    ORDER BY ticker, month_end
""")

# Drop incomplete trailing month
last_month = con.sql("SELECT max(month_end) FROM monthly").fetchone()[0]
last_day_of_data = con.sql("SELECT max(date) FROM prices").fetchone()[0]
if (last_month - last_day_of_data).days > 5:
    con.sql(f"DELETE FROM monthly WHERE month_end = DATE '{last_month}'")
    print(f"Dropped incomplete trailing month {last_month} (sample ends {last_day_of_data})")

n_months = con.sql("SELECT count(DISTINCT month_end) FROM monthly").fetchone()[0]
print(f"Monthly table: {con.sql('SELECT count(*) FROM monthly').fetchone()[0]:,} rows across {n_months} months")

# ===========================================================================
# 4) Compute monthly returns using SQL window functions
# ===========================================================================
print("\n=== 4. Window Function Return Calculations ===")

con.sql("""
    CREATE TABLE monthly_returns AS
    SELECT
        ticker,
        month_end,
        close,
        close / lag(close) OVER (PARTITION BY ticker ORDER BY month_end) - 1 AS ret
    FROM monthly
    ORDER BY ticker, month_end
""")

print(con.sql("""
    SELECT ticker, month_end, round(close, 2) AS close, round(ret * 100, 2) AS ret_pct
    FROM monthly_returns
    WHERE ticker = 'AAPL' AND month_end <= DATE '2015-05-31'
    ORDER BY month_end
""").df().to_string(index=False))

# ===========================================================================
# 5) Exploratory analytics queries
# ===========================================================================
print("\n=== 5. Database Analytics Queries ===")

print("\nExtreme monthly returns across universe:")
print(con.sql("""
    (SELECT ticker, month_end, round(ret * 100, 1) AS ret_pct FROM monthly_returns
     ORDER BY ret DESC NULLS LAST LIMIT 3)
    UNION ALL
    (SELECT ticker, month_end, round(ret * 100, 1) FROM monthly_returns
     ORDER BY ret ASC NULLS LAST LIMIT 3)
""").df().to_string(index=False))

print("\nSector performance during March 2020 market drawdown:")
con.sql("CREATE TABLE sectors (ticker VARCHAR, sector VARCHAR)")
con.executemany("INSERT INTO sectors VALUES (?, ?)", list(config.SECTORS.items()))
print(con.sql("""
    SELECT s.sector,
           count(*)                          AS stocks,
           round(avg(m.ret) * 100, 2)        AS avg_return_pct
    FROM monthly_returns m
    JOIN sectors s USING (ticker)
    WHERE m.month_end = DATE '2020-03-31'
    GROUP BY s.sector
    ORDER BY avg_return_pct DESC
""").df().to_string(index=False))

con.close()
size_mb = os.path.getsize(DB) / 1e6
csv_mb = sum(os.path.getsize(os.path.join(config.DATA_FOLDER, f))
             for f in os.listdir(config.DATA_FOLDER) if f.endswith(".csv")) / 1e6
print(f"\nStorage footprint:")
print(f"   CSV source files : {csv_mb:.1f} MB")
print(f"   DuckDB file      : {size_mb:.1f} MB ({csv_mb/size_mb:.1f}x compression ratio)")

# ===========================================================================
# CHECKPOINT
# ===========================================================================
print("\n=== CHECKPOINT ===")
con = duckdb.connect(DB, read_only=True)
assert n_tickers == 109, f"Expected 109 tickers, got {n_tickers}"
assert n_rows == 315303, f"Expected 315,303 price rows, got {n_rows:,}"
assert n_months == 138, f"Expected 138 months, got {n_months}"
assert con.sql("SELECT max(month_end) FROM monthly").fetchone()[0].isoformat() == "2026-06-30"
assert con.sql("SELECT max(length(ticker)) FROM prices").fetchone()[0] <= 5, "Extracted ticker length invalid"
assert con.sql("SELECT count(*) FROM prices WHERE close <= 0").fetchone()[0] == 0
con.close()
print(f"OK: {n_rows:,} rows, {n_tickers} tickers, {n_months} months, {size_mb:.1f} MB.")
