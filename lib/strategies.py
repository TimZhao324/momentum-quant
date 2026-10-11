"""Cross-sectional equity strategy definitions.

Each strategy provides a scoring function mapping (prices, volume, month_idx)
to cross-sectional signal scores.
"""

import config


def vol6(px, i):
    """Compute 6-month trailing return volatility for cross-sectional ranking."""
    return px.pct_change(fill_method=None).iloc[i - 5:i + 1].std()


STRATEGIES = [
    {
        "id": "MOM12_1",
        "name": "Momentum 12-1",
        "sleeve": "Momentum",
        # 12-month momentum skipping the most recent month to avoid short-term reversal
        "score": lambda px, vol, i: px.iloc[i - 1] / px.iloc[i - 12] - 1,
    },
    {
        "id": "REV1",
        "name": "1-Month Reversal",
        "sleeve": "Reversal",
        # Short-term mean reversal (prior 1-month return inverted)
        "score": lambda px, vol, i: -(px.iloc[i] / px.iloc[i - 1] - 1),
    },
    {
        "id": "TREND10",
        "name": "Above 10-Month Trend",
        "sleeve": "Trend",
        # Distance from trailing 10-month moving average
        "score": lambda px, vol, i: px.iloc[i] / px.iloc[i - 10:i + 1].mean() - 1,
    },
    {
        "id": "HIVOL6",
        "name": "High Volatility 6M",
        "sleeve": "Volatility",
        # Trailing 6-month realized return volatility
        "score": lambda px, vol, i: vol6(px, i),
    },
    {
        "id": "DVOL6",
        "name": "Dollar-Volume Momentum",
        "sleeve": "Volume",
        # 6-month growth in monthly dollar trading volume
        "score": lambda px, vol, i: (px.iloc[i] * vol.iloc[i]) / (px.iloc[i - 6] * vol.iloc[i - 6]) - 1,
    },
    {
        "id": "RAMOM6",
        "name": "Risk-Adjusted Momentum",
        "sleeve": "Quality",
        # 6-month return standardized by realized volatility
        "score": lambda px, vol, i: (px.iloc[i] / px.iloc[i - 6] - 1) / (vol6(px, i) + 1e-9),
    },
]

assert config.WARMUP >= 12, "config.WARMUP must be >= 12 (longest lookback window)"

BY_ID = {s["id"]: s for s in STRATEGIES}
