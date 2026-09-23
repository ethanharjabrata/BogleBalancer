import numpy as np
import pandas as pd
import quantstats as qs
from scipy.optimize import newton

def compute_portfolio_metrics(
    portfolio_history: pd.Series,
    prices: pd.DataFrame | None = None,
    settings: dict | None = None,
    risk_free_history: pd.Series | None = None,
    cash_flows: pd.Series | None = None,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Calculates portfolio performance metrics using QuantStats and SciPy.

    Reads benchmark and risk-free defaults from settings.yaml if not explicitly passed.

    Parameters:
    -----------
    portfolio_history : pd.Series
        Series of total portfolio net value indexed by datetime.
    prices : pd.DataFrame, optional
        DataFrame of total return prices containing the configured benchmark ticker.
    settings : dict, optional
        Summary settings containing ``benchmark_ticker`` and ``risk_free_ticker``.
    risk_free_history : pd.Series, optional
        Series of risk-free total returns (e.g., VBIL).
    cash_flows : pd.Series, optional
        Series of external cash flows (+inflows / -withdrawals).
    periods_per_year : int, default 252
        Trading periods per year.
    Returns:
    --------
    dict containing CAGR/IRR, Volatility, Sharpe Ratio, Max Drawdown, and Beta.
    """
    settings = settings or {}

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

    yearly_cagrs = _calculate_yearly_cagrs(portfolio_history)
    finite_yearly_cagrs = np.asarray(yearly_cagrs, dtype=float)
    finite_yearly_cagrs = finite_yearly_cagrs[np.isfinite(finite_yearly_cagrs)]
    if finite_yearly_cagrs.size:
        yearly_cagr_q1, yearly_cagr_q2, yearly_cagr_q3 = np.quantile(
            finite_yearly_cagrs, [0.25, 0.5, 0.75]
        )
    else:
        yearly_cagr_q1 = yearly_cagr_q2 = yearly_cagr_q3 = np.nan

    # 3. QuantStats Risk Metrics
    ann_volatility = qs.stats.volatility(port_returns, periods=periods_per_year)
    sharpe_ratio = qs.stats.sharpe(port_returns, rf=rf_rate, periods=periods_per_year)
    max_drawdown = qs.stats.max_drawdown(portfolio_history)

    # 4. Beta via QuantStats
    beta = np.nan
    benchmark_ticker = settings.get("benchmark_ticker", "VTI")
    if prices is not None and benchmark_ticker in prices:
        benchmark_history = prices[benchmark_ticker]
        bm_returns = benchmark_history.reindex(dates).pct_change().dropna()
        common_idx = port_returns.index.intersection(bm_returns.index)
        beta = qs.stats.greeks(port_returns.loc[common_idx], bm_returns.loc[common_idx])["beta"]

    return {
        return_metric_name: float(annualized_return),
        "Yearly CAGR Q1": float(yearly_cagr_q1),
        "Yearly CAGR Q2": float(yearly_cagr_q2),
        "Yearly CAGR Q3": float(yearly_cagr_q3),
        "Annualized Volatility": float(ann_volatility),
        "Sharpe Ratio": float(sharpe_ratio),
        "Max Drawdown": float(max_drawdown),
        "Beta": float(beta),
        "Benchmark Asset": settings.get("benchmark_ticker", "VTI"),
        "Risk-Free Asset": settings.get("risk_free_ticker", "VBIL"),
    }


def _calculate_yearly_cagrs(portfolio_history: pd.Series) -> list[float]:
    """Return annualized growth rates for anniversary-based yearly intervals.

    Each interval begins at the first observation and then at each preceding
    anniversary valuation. If an anniversary falls between observations, use
    the most recent value on or before it. Any remaining final interval is
    annualized using its actual elapsed time.
    """
    if len(portfolio_history) < 2:
        return []

    history = portfolio_history.sort_index().dropna()
    if len(history) < 2:
        return []

    dates = history.index
    first_date = pd.Timestamp(dates[0])
    last_date = pd.Timestamp(dates[-1])
    start_position = 0
    yearly_cagrs = []

    def interval_cagr(end_position: int) -> float:
        start_value = float(history.iloc[start_position])
        end_value = float(history.iloc[end_position])
        elapsed_days = (pd.Timestamp(dates[end_position]) - pd.Timestamp(dates[start_position])).total_seconds() / 86400
        if elapsed_days <= 0 or start_value <= 0 or end_value <= 0:
            return np.nan
        return (end_value / start_value) ** (365.25 / elapsed_days) - 1

    year_number = 1
    while True:
        anniversary = first_date + pd.DateOffset(years=year_number)
        if anniversary > last_date:
            break

        end_position = dates.searchsorted(anniversary, side="right") - 1
        if end_position > start_position:
            yearly_cagrs.append(float(interval_cagr(end_position)))
            start_position = end_position
        year_number += 1

    if start_position < len(history) - 1:
        yearly_cagrs.append(float(interval_cagr(len(history) - 1)))

    return yearly_cagrs


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
