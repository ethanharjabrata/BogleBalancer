from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import yaml

from strategies.baseline import eligible_dates, simulate_baseline
from utils.summary_statistics.calculate_summary import compute_portfolio_metrics


@pytest.fixture
def config(tmp_path):
    value = {"timeframe":{"start_year":2024,"end_year":2024}, "inflows":{"frequency":"monthly","value":1000}, "baseline":{"us_equities_ticker":"AAA","us_equities_weight":.75,"us_bonds_ticker":"BBB","us_bonds_weight":.25,"rebalance_frequency":"monthly","rebalance_threshold":.01}}
    path = tmp_path / "settings.yaml"; path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path, value


@pytest.fixture
def prices():
    dates = pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-01", "2024-02-29", "2024-03-01"])
    return pd.DataFrame({"AAA":[100, 140, 140, 180, 180], "BBB":[100, 100, 100, 100, 100]}, index=dates)


def test_outputs_are_series_and_drag_is_lower_with_taxable_dividend(config, prices):
    path, _ = config
    dividends = pd.DataFrame({"AAA":[0, 2, 0, 0, 0], "BBB":[0]*5}, index=prices.index)
    gross, net = simulate_baseline(path, market_data=prices, dividends=dividends)
    assert gross.index.equals(prices.index) and gross.dtype == float and np.isfinite(net).all()
    assert gross.iloc[-1] > net.iloc[-1]
    metrics = compute_portfolio_metrics(gross)
    assert "CAGR" in metrics


def test_summary_beta_uses_configured_benchmark_prices(config, prices):
    path, _ = config
    gross, _ = simulate_baseline(path, market_data=prices)
    metrics = compute_portfolio_metrics(
        gross,
        prices=prices,
        settings={"benchmark_ticker": "AAA"},
    )
    assert np.isfinite(metrics["Beta"])


def test_invalid_weights_fail_before_market_loading(config):
    path, value = config; value["baseline"]["us_bonds_weight"] = .20; path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ValueError, match="sum to 1"):
        simulate_baseline(path)


def test_schedule_uses_first_trading_day():
    dates = pd.to_datetime(["2024-01-31", "2024-02-02", "2024-03-01", "2024-04-01", "2025-01-02"])
    assert list(eligible_dates(dates, "monthly")) == [dates[0], dates[1], dates[2], dates[3], dates[4]]
    assert list(eligible_dates(dates, "quarterly")) == [dates[0], dates[3], dates[4]]
    assert list(eligible_dates(dates, "yearly")) == [dates[0], dates[4]]


def test_threshold_and_invalid_market_data(config, prices):
    path, value = config
    high = deepcopy(value); high["baseline"]["rebalance_threshold"] = 99; high["inflows"]["frequency"] = "yearly"; path.write_text(yaml.safe_dump(high), encoding="utf-8")
    gross, _ = simulate_baseline(path, market_data=prices)
    # Without rebalancing only the first contribution exists: 75% AAA, 25% BBB.
    assert gross.iloc[-1] == pytest.approx(1600)
    with pytest.raises(ValueError, match="complete"):
        simulate_baseline(path, market_data=prices.drop(columns="BBB"))
    high["inflows"]["frequency"] = "weekly"; path.write_text(yaml.safe_dump(high), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported inflow frequency"):
        simulate_baseline(path, market_data=prices)
