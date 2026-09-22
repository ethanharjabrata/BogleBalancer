"""Run with: .venv\\Scripts\\python -m examples.baseline"""

from pathlib import Path

import yaml

from strategies.baseline import load_market_data, load_settings, simulate_baseline
from utils.summary_statistics.calculate_summary import compute_portfolio_metrics

BASELINE_SETTINGS_PATH = Path(__file__).parents[1] / "strategies" / "settings.yaml"
SUMMARY_SETTINGS_PATH = Path(__file__).with_name("settings.yaml")


def load_summary_settings() -> dict:
    with SUMMARY_SETTINGS_PATH.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def main() -> None:
    baseline_settings = load_settings(BASELINE_SETTINGS_PATH)
    summary_settings = load_summary_settings()
    baseline_config = baseline_settings["baseline"]
    tickers = (
        baseline_config["us_equities_ticker"],
        baseline_config["us_bonds_ticker"],
    )
    prices, dividends = load_market_data(
        tickers,
        baseline_settings["timeframe"]["start_year"],
        baseline_settings["timeframe"]["end_year"],
    )
    gross, net = simulate_baseline(
        BASELINE_SETTINGS_PATH,
        market_data=prices,
        dividends=dividends,
    )
    if gross.empty or net.empty or gross.iloc[-1] < net.iloc[-1]:
        raise RuntimeError("Gross ending value must be greater than or equal to net ending value")
    for label, history in (("Gross / no drag", gross), ("Net / drag-adjusted", net)):
        print(f"\n{label}")
        for metric, value in compute_portfolio_metrics(
            history,
            prices=prices,
            settings=summary_settings,
        ).items():
            print(f"{metric}: {value}")
        print(f"Ending value: {history.iloc[-1]:,.2f}")


if __name__ == "__main__":
    main()
