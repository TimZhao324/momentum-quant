"""Optional utility: Ingest and update price history CSVs from Yahoo Finance.

Usage:
    .venv/bin/python tools/download_data.py            # all tickers
    .venv/bin/python tools/download_data.py AAPL SPY   # selected tickers
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "lib")]
os.chdir(ROOT)

import pandas as pd
import yfinance as yf

import config

RETRIES = 3
PAUSE_BETWEEN = 1.5
PAUSE_ON_FAIL = 5.0


def download_one(ticker, start=config.START_DATE):
    """Download daily price series for a single ticker."""
    data = yf.download(ticker, start=start, interval="1d",
                       auto_adjust=True,
                       progress=False, threads=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    keep = [c for c in ("Close", "High", "Low", "Open", "Volume") if c in data.columns]
    return data[keep]


def main(tickers):
    """Download each ticker sequentially, persisting to disk only on successful retrieval."""
    os.makedirs(config.DATA_FOLDER, exist_ok=True)
    ok, failed = [], []
    for n, ticker in enumerate(tickers, 1):
        print(f"[{n}/{len(tickers)}] {ticker} ... ", end="", flush=True)
        data = None
        for attempt in range(1, RETRIES + 1):
            try:
                data = download_one(ticker)
                if len(data) > 0:
                    break
                data = None
            except Exception as error:
                print(f"\n    attempt {attempt}: {type(error).__name__}: {error}")
            time.sleep(PAUSE_ON_FAIL)

        if data is None or len(data) == 0:
            print("NO DATA (skipped, existing file untouched)")
            failed.append(ticker)
        else:
            path = os.path.join(config.DATA_FOLDER, ticker + ".csv")
            data.to_csv(path)
            print(f"{len(data)} rows -> {path}")
            ok.append(ticker)
        time.sleep(PAUSE_BETWEEN)

    print(f"\n{len(ok)} updated, {len(failed)} failed.")
    if failed:
        print("Failed tickers: " + " ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    requested = sys.argv[1:] or config.TICKERS
    print(f"Downloading {len(requested)} tickers from {config.START_DATE}.")
    sys.exit(main(requested))
