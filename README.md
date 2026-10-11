# Does 12-Month Momentum Pick Stocks — or Just Pick Beta?

A quantitative equity research study evaluating cross-sectional momentum across 109 US large-cap equities over 138 months (2015–2026), measured via three independent methodologies: portfolio simulation, Information Coefficient (Rank IC), and multi-factor regression decomposition.

**Deliverable**: The apparent outperformance of cross-sectional momentum in large-cap equities is primarily explained by market beta and growth factor loadings rather than stock selection skill.

---

## Key Findings

| Metric | Long-Only Top-10 | Long-Short Deciles | SPY Benchmark |
|---|---|---|---|
| Growth of $1 | **$11.71** | $1.36 | $4.57 |
| CAGR | **26.64%** | 2.98% | 15.71% |
| Sharpe Ratio (rf = 0) | **1.23** | 0.29 | 1.05 |
| Max Drawdown | −21.7% | −32.1% | −23.9% |
| Beta to SPY | **1.10** | **−0.07** | 1.00 |

While the unhedged long-only portfolio appears strong, rigorous quantitative decomposition reveals:

1. **Rank IC = +0.0078 (t = 0.39)**: Month-by-month cross-sectional ranking predictive power is statistically indistinguishable from noise.
2. **Quantile Monotonicity Fails**: Partitioning into quintile buckets does not yield a monotonic return profile; Q5 (worst rank) outperforms intermediate quintiles.
3. **Factor Attribution (R² = 0.65)**: Two-thirds of long-only variance is explained by standard equity factors (market beta 1.01). Removing market beta in the dollar-neutral implementation eliminates 90% of returns, leaving a growth tilt rather than alpha (t = 0.37).
4. **Portfolio Information Ratio = 0.78**: Active return of +10.15%/yr against tracking error of 13.05% (t = 2.51), subject to survivorship and lookback caveats.

Detailed analysis: **[RESULTS.md](RESULTS.md)**.

---

## Methodological Caveats

1. **Survivorship Bias**: Universe reflects S&P-100 constituents backfilled to 2015. Acquired, merged, or delisted firms are excluded.
2. **Risk-Free Rate**: Metrics assume zero risk-free rate. Actual excess-return Sharpe ratios would adjust downwards reflecting prevailing short-term yields.
3. **Execution & Financing**: Transaction costs modeled at 10 bps per unit turnover; short borrow costs evaluated at 0 to 300 bps/yr.

---

## Quickstart

```bash
git clone <repo-url> momentum-quant && cd momentum-quant
bash setup.sh                        # Provision venv and dependencies
.venv/bin/python tools/pipeline.py   # Run backtests, generate charts and JSON
.venv/bin/python tools/serve.py      # Serve dashboard at http://localhost:8000
.venv/bin/python steps/check.py      # Run automated acceptance test suite
```

All 109 price histories are committed under `data/`, ensuring 100% offline reproducibility across environments.

---

## Pipeline Architecture

```
data/*.csv  ──►  lib/timlib.py     Ingestion, resampling, returns, NAV, metrics
                 lib/strategies.py Cross-sectional signal scoring definitions
                 lib/engine.py     Simulation engine (long-only & dollar-neutral)
                 lib/analysis.py   Rank IC, quintiles, OLS regression, VIF
                      │
                      ├──►  steps/01..07        Modular analysis workflows
                      └──►  tools/pipeline.py ──►  web/data/*.json + web/charts/*.png
                                                        │
                                                        └──►  web/index.html
```

### Module Overview

- [`steps/00_hello_server.sh`](steps/00_hello_server.sh): Hardware, OS, toolchain and environment audit.
- [`steps/01_load_and_plot.py`](steps/01_load_and_plot.py): Price data loading, vectorised returns, distribution plots.
- [`steps/02_metrics.py`](steps/02_metrics.py): NAV compounding, CAGR, annualized volatility, Sharpe, max drawdown.
- [`steps/03_backtest.py`](steps/03_backtest.py): Portfolio execution engine, look-ahead bias demonstration, turnover drag.
- [`steps/04_ic_and_ir.py`](steps/04_ic_and_ir.py): Parametric vs Rank IC, t-statistics, Grinold's law, quantile testing.
- [`steps/05_multifactor.py`](steps/05_multifactor.py): Linear regression attribution, basis change, VIF collinearity diagnostics.
- [`steps/06_database_load.py`](steps/06_database_load.py): DuckDB columnar ingestion, schema definition, SQL window functions.
- [`steps/07_sql_vs_pandas.py`](steps/07_sql_vs_pandas.py): Cross-engine numerical verification between Pandas and DuckDB.
- [`steps/check.py`](steps/check.py): Comprehensive test suite verifying 71 quantitative assertions.

---

## Tech Stack

- **Runtime**: Linux / Python 3.12
- **Data & Analytics**: Pandas, NumPy, DuckDB, PyArrow
- **Visualization**: Matplotlib
- **Web Dashboard**: Vanilla HTML5, CSS3, JavaScript (no external runtime dependencies)
