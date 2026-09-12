from pathlib import Path
import numpy as np
import pandas as pd
import quantstats as qs
import yaml
from scipy.optimize import newton

# Resolve path relative to this script's directory
SETTINGS_PATH = Path(__file__).parent / "settings.yaml"


def load_settings(config_path: Path = SETTINGS_PATH) -> dict:
    """Loads configuration settings from YAML."""
    if not config_path.exists():
        return {"benchmark_ticker": "VTI", "risk_free_ticker": "VBIL"}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def compute_portfolio_metrics(
    portfolio_history: pd.Series,
    benchmark_history: pd.Series | None = None,
    risk_free_history: pd.Series | None = None,
    cash_flows: pd.Series | None = None,
    periods_per_year: int = 252,
    config_path: Path = SETTINGS_PATH,
) -> dict[str, float]:
    """Calculates portfolio performance metrics using QuantStats and SciPy.

    Reads benchmark and risk-free defaults from settings.yaml if not explicitly passed.

    Parameters:
    -----------
    portfolio_history : pd.Series
        Series of total portfolio net value indexed by datetime.
    benchmark_history : pd.Series, optional
        Series of benchmark total return prices (e.g., VTI) for Beta.
    risk_free_history : pd.Series, optional
        Series of risk-free total returns (e.g., VBIL).
    cash_flows : pd.Series, optional
        Series of external cash flows (+inflows / -withdrawals).
    periods_per_year : int, default 252
        Trading periods per year.
    config_path : Path
        Path to settings.yaml configuration file.

    Returns:
    --------
    dict containing CAGR/IRR, Volatility, Sharpe Ratio, Max Drawdown, and Beta.
    """
    settings = load_settings(config_path)

    portfolio_history = portfolio_history.sort_index().dropna()
    dates = portfolio_history.index
    port_returns = portfolio_history.pct_change().dropna()

    # 1. Risk-Free Rate Handling
    rf_rate = 0.0
    if risk_free_history is not None:
        rf_returns = risk_free_history.reindex(dates).pct_change().dropna()
        if not rf_returns.empty:
            rf_rate = (1 + rf_returns.mean()) ** periods_per_year - 1
        else:
            rf_rate = 0.0  # Fallback to zero risk-free rate if history is missing

    # 2. Return Metric (IRR vs CAGR via QuantStats)
    if cash_flows is not None and not cash_flows.dropna().empty:
        cf_dates = [dates[0]]
        cf_values = [-float(portfolio_history.iloc[0])]

        aligned_cf = cash_flows.reindex(dates).fillna(0.0)
        for d, cf_val in aligned_cf.items():
            if cf_val != 0 and d != dates[0] and d != dates[-1]:
                cf_dates.append(d)
                cf_values.append(-float(cf_val))

        cf_dates.append(dates[-1])
        cf_values.append(float(portfolio_history.iloc[-1]))

        annualized_return = _calculate_xirr_scipy(pd.Series(cf_values, index=cf_dates))
        return_metric_name = "Annualized IRR (Cash-Flow Adj.)"
    else:
        annualized_return = qs.stats.cagr(portfolio_history, rf=rf_rate, periods=periods_per_year)
        return_metric_name = "CAGR"

    # 3. QuantStats Risk Metrics
    ann_volatility = qs.stats.volatility(port_returns, periods=periods_per_year)
    sharpe_ratio = qs.stats.sharpe(port_returns, rf=rf_rate, periods=periods_per_year)
    max_drawdown = qs.stats.max_drawdown(portfolio_history)

    # 4. Beta via QuantStats
    beta = np.nan
    if benchmark_history is not None:
        bm_returns = benchmark_history.reindex(dates).pct_change().dropna()
        common_idx = port_returns.index.intersection(bm_returns.index)
        beta = qs.stats.greeks(port_returns.loc[common_idx], bm_returns.loc[common_idx])["beta"]

    return {
        return_metric_name: float(annualized_return),
        "Annualized Volatility": float(ann_volatility),
        "Sharpe Ratio": float(sharpe_ratio),
        "Max Drawdown": float(max_drawdown),
        "Beta": float(beta),
        "Benchmark Asset": settings.get("benchmark_ticker", "VTI"),
        "Risk-Free Asset": settings.get("risk_free_ticker", "VBIL"),
    }


def _calculate_xirr_scipy(cash_flows: pd.Series) -> float:
    """Calculates XIRR using SciPy's Newton-Raphson solver."""
    dates = cash_flows.index
    values = cash_flows.values
    t0 = dates[0]
    years = np.array([(d - t0).days / 365.25 for d in dates])

    def npv(r: float) -> float:
        return np.sum(values / ((1 + r) ** years))

    def npv_prime(r: float) -> float:
        return np.sum(-years * values / ((1 + r) ** (years + 1)))

    try:
        return float(newton(npv, x0=0.10, fprime=npv_prime, maxiter=100))
    except (RuntimeError, ZeroDivisionError):
        return np.nan