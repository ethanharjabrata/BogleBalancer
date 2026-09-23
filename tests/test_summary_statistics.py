import numpy as np
import pandas as pd
import pytest

from utils.summary_statistics.calculate_summary import compute_portfolio_metrics


def test_yearly_cagr_quartiles_include_annualized_final_partial_year():
    dates = pd.to_datetime(
        ["2021-01-01", "2022-01-01", "2023-01-01", "2023-07-01"]
    )
    history = pd.Series([100.0, 110.0, 99.0, 104.0], index=dates)

    metrics = compute_portfolio_metrics(history)
    interval_days = np.diff(dates.asi8) / (24 * 60 * 60 * 1e9)
    yearly_cagrs = np.array(
        [
            1.1 ** (365.25 / interval_days[0]) - 1,
            0.9 ** (365.25 / interval_days[1]) - 1,
            (104.0 / 99.0) ** (365.25 / interval_days[2]) - 1,
        ]
    )

    for label, expected in zip(
        ("Yearly CAGR Q1", "Yearly CAGR Q2", "Yearly CAGR Q3"),
        np.quantile(yearly_cagrs, [0.25, 0.5, 0.75]),
    ):
        assert metrics[label] == pytest.approx(expected)
