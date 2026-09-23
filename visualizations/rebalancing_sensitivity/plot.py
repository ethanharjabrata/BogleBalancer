"""Compare baseline IRR quartiles across rebalancing thresholds.

Run from the repository root with ``python -m
visualizations.rebalancing_sensitivity.plot``. Outputs are written beside this
module, regardless of the current working directory.
"""
from __future__ import annotations

from decimal import Decimal
from math import isfinite
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import yaml

REBALANCING_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = REBALANCING_DIR.parents[1]
SETTINGS_PATH = REBALANCING_DIR / "settings.yaml"
STRATEGY_SETTINGS_PATH = REPOSITORY_ROOT / "strategies" / "settings.yaml"
RESULTS_PATH = REBALANCING_DIR / "rebalancing_sensitivity_results.csv"

# Support direct execution from outside the repository as well as ``python -m``.
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from strategies.baseline import (  # noqa: E402
    _validate_config,
    load_market_data,
    load_settings,
    simulate_baseline,
)
from utils.summary_statistics.calculate_summary import compute_portfolio_metrics  # noqa: E402

QUARTILE_COLUMNS = ("Yearly IRR Q1", "Yearly IRR Q2", "Yearly IRR Q3")


def load_sensitivity_settings(path: Path = SETTINGS_PATH) -> dict[str, Any]:
    """Load and validate the sensitivity range and optional summary settings."""
    try:
        with path.open(encoding="utf-8") as stream:
            settings = yaml.safe_load(stream)
    except FileNotFoundError as exc:
        raise ValueError(f"Sensitivity settings file does not exist: {path}") from exc
    if not isinstance(settings, dict):
        raise ValueError("Sensitivity settings must be a YAML mapping")

    values: dict[str, float] = {}
    for key in (
        "rebalance_threshold_min",
        "rebalance_threshold_max",
        "rebalance_threshold_step",
    ):
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a finite numeric value")
        number = float(value)
        if not isfinite(number):
            raise ValueError(f"{key} must be a finite numeric value")
        values[key] = number

    minimum = values["rebalance_threshold_min"]
    maximum = values["rebalance_threshold_max"]
    step = values["rebalance_threshold_step"]
    if not 0 <= minimum <= maximum <= 1:
        raise ValueError(
            "rebalance thresholds must satisfy 0 <= min <= max <= 1"
        )
    if step <= 0:
        raise ValueError("rebalance_threshold_step must be greater than zero")
    return settings


def build_thresholds(settings: dict[str, Any]) -> list[float]:
    """Build the inclusive step grid without floating point accumulation."""
    minimum = Decimal(str(settings["rebalance_threshold_min"]))
    maximum = Decimal(str(settings["rebalance_threshold_max"]))
    step = Decimal(str(settings["rebalance_threshold_step"]))

    thresholds: list[float] = []
    seen: set[float] = set()
    index = 0
    while True:
        value = minimum + index * step
        if value > maximum:
            break
        threshold = float(value)
        if threshold in seen:
            raise ValueError(
                "rebalance_threshold_step is too small to produce distinct numeric thresholds"
            )
        thresholds.append(threshold)
        seen.add(threshold)
        index += 1
    if not thresholds:
        raise ValueError("Sensitivity settings produced no rebalance thresholds")
    return thresholds


def _collect_results(
    thresholds: list[float],
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    strategy_settings: dict[str, Any],
    sensitivity_settings: dict[str, Any],
) -> pd.DataFrame:
    inflows = strategy_settings["inflows"]
    rows: list[dict[str, float | str]] = []
    for threshold in thresholds:
        gross_history, net_history = simulate_baseline(
            STRATEGY_SETTINGS_PATH,
            rebalance_threshold=threshold,
            market_data=prices,
            dividends=dividends,
        )
        for path_name, history in (("gross", gross_history), ("net", net_history)):
            metrics = compute_portfolio_metrics(
                history,
                prices=prices,
                settings=sensitivity_settings,
                inflow_value=float(inflows["value"]),
                inflow_frequency=inflows["frequency"],
            )
            quartiles = [float(metrics[key]) for key in QUARTILE_COLUMNS]
            if not all(isfinite(value) for value in quartiles):
                raise ValueError(
                    f"Threshold {threshold:g} ({path_name}) has no finite yearly IRR quartiles"
                )
            if not quartiles[0] <= quartiles[1] <= quartiles[2]:
                raise ValueError(
                    f"Threshold {threshold:g} ({path_name}) has unordered yearly IRR quartiles"
                )
            rows.append(
                {
                    "rebalance_threshold": threshold,
                    "path": path_name,
                    **dict(zip(QUARTILE_COLUMNS, quartiles)),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["rebalance_threshold", "path", *QUARTILE_COLUMNS],
    )


def _save_plot(results: pd.DataFrame) -> Path:
    """Save gross and net yearly IRR quartiles together in one figure."""
    fig, ax = plt.subplots()
    try:
        for path_name, label in (
            ("gross", "Gross / no drag"),
            ("net", "Net / drag-adjusted"),
        ):
            selected = results.loc[results["path"] == path_name].sort_values(
                "rebalance_threshold"
            )
            x = selected["rebalance_threshold"].to_numpy(dtype=float)
            q1 = selected["Yearly IRR Q1"].to_numpy(dtype=float)
            q2 = selected["Yearly IRR Q2"].to_numpy(dtype=float)
            q3 = selected["Yearly IRR Q3"].to_numpy(dtype=float)
            quartiles_are_finite = all(
                np.isfinite(values).all() for values in (x, q1, q2, q3)
            )
            if not quartiles_are_finite:
                raise ValueError(
                    f"Cannot plot non-finite yearly IRR values for {path_name}"
                )
            if not ((q1 <= q2).all() and (q2 <= q3).all()):
                raise ValueError(
                    f"Cannot plot unordered yearly IRR quartiles for {path_name}"
                )
            ax.errorbar(
                x,
                q2,
                yerr=np.vstack((q2 - q1, q3 - q2)),
                fmt="o-",
                capsize=4,
                label=label,
            )

        ax.set_title("Yearly IRR by Rebalance Threshold")
        ax.set_xlabel("Rebalance threshold (maximum absolute asset-weight drift)")
        ax.set_ylabel("Yearly IRR")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        output = REBALANCING_DIR / "irr_vs_rebalance_threshold.png"
        fig.savefig(output, dpi=150)
        return output
    finally:
        plt.close(fig)


def main() -> None:
    """Run the configured threshold sweep and write its table and plots."""
    sensitivity_settings = load_sensitivity_settings()
    thresholds = build_thresholds(sensitivity_settings)

    # Validate strategy configuration before making a market-data request.
    strategy_settings = load_settings(STRATEGY_SETTINGS_PATH)
    strategy_config = _validate_config(
        strategy_settings, rebalance_threshold=thresholds[0]
    )
    prices, dividends = load_market_data(
        strategy_config.tickers,
        strategy_config.start_year,
        strategy_config.end_year,
    )

    results = _collect_results(
        thresholds, prices, dividends, strategy_settings, sensitivity_settings
    )
    results.to_csv(RESULTS_PATH, index=False)
    plot = _save_plot(results)
    print(f"Saved {len(results)} rows to {RESULTS_PATH}")
    print(f"Saved plot to {plot}")


if __name__ == "__main__":
    main()
