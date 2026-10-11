# Quantitative Research Report: Cross-Sectional Momentum in US Large Caps

**Key Conclusion**: A concentrated long-only momentum strategy outperformed the S&P 500 by +10.93%/year annualized over 2015–2026. However, three independent empirical tests confirm that this apparent performance is attributed to market beta exposure and growth style tilts rather than cross-sectional stock selection capability.

---

## Methodology & Experimental Design

| Parameter | Specification |
|---|---|
| **Universe** | 109 tickers (100 S&P-100 constituents + 9 macro/factor proxy ETFs) |
| **Sample Window** | 138 monthly intervals (Jan 2015 – Jun 2026); 125-month backtest window post 12-month warmup |
| **Primary Signal** | Momentum 12-1: Trailing 12-month return skipping the most recent month ($t-12$ to $t-1$) |
| **Portfolios** | (a) Long-Only: Top 10 equal-weighted; (b) Long-Short: Top decile long vs bottom decile short, dollar-neutral |
| **Rebalance Frequency** | Monthly, at month-end closing prices |
| **Friction Model** | 10 bps per unit turnover (5 bps commission + 5 bps slippage); 100 bps/yr short borrow financing |
| **Look-Ahead Controls** | Strict execution separation: signal formed at $t$, forward return earned across $t+1$ |

Five additional signals (1-month reversal, 10-month trend, 6-month high volatility, dollar-volume momentum, risk-adjusted momentum) were evaluated under identical conditions.

---

## Empirical Finding 1: Portfolio Simulation Disconnect

| Metric | Long-Only Top-10 | Long-Short Dollar-Neutral | SPY Benchmark |
|---|---|---|---|
| **Growth of $1** | **$11.71** | $1.36 | $4.57 |
| **CAGR** | **26.64%** | 2.98% | 15.71% |
| **Annual Volatility** | 21.14% | 12.68% | 15.13% |
| **Sharpe Ratio (rf = 0)** | **1.23** | 0.29 | 1.05 |
| **Maximum Drawdown** | −21.66% | −32.10% | −23.93% |
| **Beta to SPY** | 1.10 | **−0.07** | 1.00 |
| **Monthly Turnover** | 0.55 | 0.57 | — |
| **Transaction Cost Drag** | 66 bps/yr | 69 bps/yr | — |

- **Portfolio Information Ratio**: 0.78 (active return +10.15%/yr vs tracking error 13.05%, $t = 2.51$).
- **Beta Neutralisation Impact**: When market exposure is removed by taking offsetting short positions in bottom-decile names, portfolio CAGR collapses from 26.64% to 2.98%. Beta drops from 1.10 to −0.07, taking ~90% of returns with it.

---

## Empirical Finding 2: Cross-Sectional Predictive Power (IC & Monotonicity)

Cross-sectional rank Information Coefficient (Spearman correlation between period-$t$ scores and period-$(t+1)$ forward returns):

| Signal Identifier | Mean Rank IC | IC Std | Signal IR | t-Statistic | Statistical Verdict |
|---|---|---|---|---|---|
| **Momentum 12-1** | +0.0078 | 0.227 | 0.035 | **+0.39** | Noise ($p = 0.70$) |
| **1-Month Reversal** | −0.0009 | 0.193 | −0.005 | −0.05 | Noise |
| **10-Month Trend** | +0.0157 | 0.230 | 0.068 | +0.76 | Noise |
| **High Volatility 6M** | **+0.0407** | 0.204 | 0.199 | **+2.23** | Statistically Significant |
| **Dollar-Volume Momentum** | +0.0041 | 0.142 | 0.029 | +0.32 | Noise |
| **Risk-Adjusted Momentum** | +0.0103 | 0.211 | 0.049 | +0.54 | Noise |

Key diagnostics:
1. **Hypothesis Test**: With $t = 0.39$, the null hypothesis that Momentum 12-1 possesses zero predictive power cannot be rejected.
2. **Quantile Monotonicity Breakdown**: Sorting into score quintiles yields average next-month returns of: Q1 = 1.77%, Q2 = 1.17%, Q3 = 1.30%, Q4 = 1.20%, Q5 = 1.39%. The bottom quintile (Q5) outperforms Q2, Q3, and Q4. The Q1−Q5 return spread of 4.51%/yr exhibits $t = 0.89$ (insignificant).
3. **Parametric Distortion**: Pearson IC (+0.0430) is 5.5× larger than Rank IC (+0.0078), distorted by extreme outliers rather than consistent cross-sectional ordering.
4. **Fundamental Law Verification**: Grinold's law predicts $IR \approx IC \times \sqrt{\text{breadth}} = 0.0078 \times \sqrt{1308} = 0.28$, aligning with measured dollar-neutral Sharpe of 0.29.

---

## Empirical Finding 3: Multi-Factor Risk Attribution

Six-factor linear regression ($n = 125$ monthly periods):
$$R_{t} = \alpha + \beta_{\text{MKT}} R_{\text{MKT}} + \beta_{\text{GROWTH}} R_{\text{GROWTH}} + \beta_{\text{SIZE}} R_{\text{SIZE}} + \beta_{\text{RATES}} R_{\text{RATES}} + \beta_{\text{GOLD}} R_{\text{GOLD}} + \beta_{\text{USD}} R_{\text{USD}} + \varepsilon_t$$

| Factor Loading | Long-Only Portfolio | Long-Short Dollar-Neutral |
|---|---|---|
| **Annualized Alpha ($\alpha$)** | +8.45% ($t = 1.83$) | +1.65% (**$t = 0.37$**, insignificant) |
| **MKT (Market)** | 1.013 ($t = 11.3$) | −0.075 ($t = −0.87$) |
| **GROWTH (QQQ − SPY)** | 0.439 ($t = 2.64$) | **0.431 ($t = 2.71$, significant)** |
| **Macro Factors (SIZE, RATES, GOLD, USD)** | All $|t| < 2.0$ | All $|t| < 2.0$ |
| **Coefficient of Determination ($R^2$)** | **0.650** | 0.104 |

Two-thirds of long-only portfolio variance is explained by standard factor exposures. While the long-short book successfully achieves market neutrality, it carries significant growth loading ($\beta = 0.43, t = 2.71$). True alpha is indistinguishable from zero.

### Parameterisation & Multicollinearity Analysis

Regressing against raw ETF returns (SPY, QQQ, IWM) yields an identical $R^2$ (0.640) and identical alpha (7.64%/yr), but exhibits variance inflation factors (VIF) of 12.2, 6.7, and 4.3 due to high mutual correlation ($\rho \approx 0.90$). The resulting market coefficient standard error is 3.2× wider (0.266 vs 0.083). Orthogonalized spread factors reduce all VIFs below 1.7 while preserving identical explanatory power.

---

## Limitations & Robustness Considerations

1. **Survivorship Bias**: Survivorship in current constituent universe creates systematic upward return drift.
2. **Sector Neutrality**: Unconstrained cross-sectional ranking produces structural tech overweighting during strong tech cycles.
3. **Risk-Free Rate Benchmarking**: Assumed zero risk-free rate; subtracting cash returns reduces all Sharpe ratios by ~0.20–0.30.
4. **Sample Specificity**: Mega-cap equities over a tech-dominant bull market (2015–2026) may differ from small-cap and international momentum regimes.
5. **Multiple Hypothesis Testing**: High Volatility 6M emerged as the sole signal clearing $|t| > 2$ ($IC = +0.0407, t = 2.23$), requiring adjustment for multiple testing before practical deployment.

---

## Reproduction Instructions

```bash
bash setup.sh
.venv/bin/python tools/pipeline.py
.venv/bin/python steps/check.py      # Verifies 71 quantitative assertions
```
