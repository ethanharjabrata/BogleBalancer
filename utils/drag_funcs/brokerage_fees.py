from pathlib import Path
from typing import Any

import yaml

SETTINGS_PATH = Path(__file__).with_name("settings.yaml")


def _configured_function() -> str:
	with SETTINGS_PATH.open(encoding="utf-8") as settings_file:
		settings = yaml.safe_load(settings_file) or {}
	return settings.get("brokerage_fee_func", "us_investor")


def _us_investor(params: dict[str, Any]) -> tuple[float, float, float, float]:
	value = float(params.get("value", params.get("amount", 0.0)))
	return value, value, value, 0.0


def calculate_brokerage_fees(params: dict[str, Any]) -> tuple[float, float, float, float]:
	"""Return brokerage-adjusted value as (original, post-fee, original, fees)."""
	functions = {"us_investor": _us_investor}
	function_name = _configured_function()
	try:
		function = functions[function_name]
	except KeyError as error:
		raise ValueError(f"Unknown brokerage fee function: {function_name}") from error
	return function(params)
