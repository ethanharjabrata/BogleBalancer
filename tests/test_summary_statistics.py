import numpy as np
import pandas as pd
import pytest

from utils.summary_statistics.calculate_summary import compute_portfolio_metrics


def test_yearly_irr_quartiles_include_annualized_final_partial_year():
    dates = pd.to_datetime(
        ["2021-01-01", "2022-01-01", "2023-01-01", "2023-07-01"]
    )
    history = pd.Series([100.0, 110.0, 99.0, 104.0], index=dates)

    metrics = compute_portfolio_metrics(history)
    interval_days = np.diff(dates.asi8) / (24 * 60 * 60 * 1e9)
    yearly_irrs = np.array(
        [
            1.1 ** (365.25 / interval_days[0]) - 1,
            0.9 ** (365.25 / interval_days[1]) - 1,
            (104.0 / 99.0) ** (365.25 / interval_days[2]) - 1,
        ]
    )

    for label, expected in zip(
        ("Yearly IRR Q1", "Yearly IRR Q2", "Yearly IRR Q3"),
        np.quantile(yearly_irrs, [0.25, 0.5, 0.75]),
    ):
        assert metrics[label] == pytest.approx(expected)


def test_regular_contributions_are_removed_from_annualized_returns():
    dates = pd.date_range("2020-01-01", "2022-01-01", freq="MS")
    dates = dates.insert(0, pd.Timestamp("2019-12-01"))
    history = pd.Series(1000.0 * np.arange(1, len(dates) + 1), index=dates)

    metrics = compute_portfolio_metrics(
        history,
        inflow_value=1000.0,
        inflow_frequency="monthly",
    )

    assert metrics["Annualized IRR (Cash-Flow Adj.)"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["Yearly IRR Q2"] == pytest.approx(0.0, abs=1e-6)
