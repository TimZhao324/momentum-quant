"""Configuration parameters for the quantitative equity research framework.

Defines the stock universe, risk factor proxies, backtest parameters,
and execution cost assumptions.
"""

# ---------------------------------------------------------------------------
# 1) Stock Universe (S&P 100 constituents grouped by GICS sector)
# ---------------------------------------------------------------------------
UNIVERSE_BY_SECTOR = {
    "Information Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE",
                               "CSCO", "ACN", "AMD", "INTC", "IBM", "QCOM", "TXN", "NOW"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ",
                               "TMUS", "CHTR"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX",
                               "BKNG", "TGT", "GM", "F"],
    "Consumer Staples":       ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ",
                               "CL", "KHC"],
    "Health Care":            ["UNH", "JNJ", "LLY", "MRK", "ABBV", "PFE", "TMO", "ABT",
                               "DHR", "BMY", "AMGN", "GILD", "CVS", "MDT"],
    "Financials":             ["JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "C",
                               "BLK", "SCHW", "USB", "COF", "PYPL"],
    "Industrials":            ["HON", "CAT", "BA", "GE", "UPS", "RTX", "UNP", "LMT",
                               "DE", "MMM", "EMR", "FDX", "GD"],
    "Energy":                 ["XOM", "CVX", "COP", "SLB"],
    "Materials":              ["SHW", "APD", "NEM"],
    "Utilities":              ["NEE", "DUK", "SO", "AEP", "EXC"],
    "Real Estate":            ["AMT", "SPG"],
}

STOCKS = [t for names in UNIVERSE_BY_SECTOR.values() for t in names]
SECTORS = {t: sec for sec, names in UNIVERSE_BY_SECTOR.items() for t in names}

STOCK_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon",
    "NVDA": "NVIDIA", "TSLA": "Tesla", "META": "Meta", "JPM": "JPMorgan",
    "KO": "Coca-Cola", "V": "Visa", "MA": "Mastercard", "UNH": "UnitedHealth",
    "LLY": "Eli Lilly", "JNJ": "Johnson & Johnson", "XOM": "Exxon Mobil",
    "HD": "Home Depot", "COST": "Costco", "WMT": "Walmart", "NFLX": "Netflix",
    "AVGO": "Broadcom", "SPY": "S&P 500 ETF",
}

BENCHMARK = "SPY"

# ---------------------------------------------------------------------------
# 2) Risk Factor Proxies (ETFs) & Multi-Factor Model Specification
# ---------------------------------------------------------------------------
FACTOR_ETFS = {
    "SPY": "Market", "QQQ": "Growth", "IWM": "Small Cap", "TLT": "Rates",
    "HYG": "Credit", "LQD": "IG Credit", "GLD": "Gold", "DBC": "Commodities",
    "UUP": "Dollar",
}

MARKET_ETF = "SPY"
SPREAD_FACTORS = {"QQQ", "IWM"}    # Formulated as spread excess returns (ETF - SPY)

# Factors used in linear cross-sectional & time-series regressions
REGRESSION_FACTORS = [
    ("MKT",    "SPY", None),    # Broad equity market excess return
    ("GROWTH", "QQQ", "SPY"),   # QQQ - SPY (Growth tilt spread)
    ("SIZE",   "IWM", "SPY"),   # IWM - SPY (Small-cap style spread)
    ("RATES",  "TLT", None),    # Long-term US Treasury bond return
    ("GOLD",   "GLD", None),    # Gold commodity proxy
    ("USD",    "UUP", None),    # US Dollar index proxy
]

TICKERS = STOCKS + [etf for etf in FACTOR_ETFS if etf not in STOCKS]

# ---------------------------------------------------------------------------
# 3) Directory Paths & Sample Periods
# ---------------------------------------------------------------------------
DATA_FOLDER = "data"
WEB_DATA_FOLDER = "web/data"
CHARTS_FOLDER = "web/charts"
DUCKDB_FILE = "data/tim.duckdb"
START_DATE = "2015-01-01"

# ---------------------------------------------------------------------------
# 4) Transaction Costs & Financing Assumptions (Basis Points)
# ---------------------------------------------------------------------------
COMMISSION_BPS = 5           # Execution commission per unit turnover
SLIPPAGE_BPS = 5             # Price impact / slippage assumption
BORROW_BPS_PER_YEAR = 100    # Annualized short borrowing cost (~100 bps/yr)

# ---------------------------------------------------------------------------
# 5) Portfolio & Signal Backtest Configuration
# ---------------------------------------------------------------------------
TOP_N = 10                   # Long-only top quintile/decile holdings count
DECILE = 10                  # Quantile partition parameter for long-short decile portfolios
WARMUP = 12                  # Signal lookback window (months)
MIN_NAMES_FOR_IC = 20        # Minimum cross-sectional coverage for Rank IC evaluation

# ---------------------------------------------------------------------------
# 6) Risk-Free Rate
# ---------------------------------------------------------------------------
RISK_FREE_RATE = 0.0         # Benchmark risk-free rate assumption (raw returns)
