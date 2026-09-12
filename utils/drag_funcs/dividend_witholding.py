from pathlib import Path
from typing import Any

import yaml

from .capital_gains import TaxProfile, calculate_niit_tax

SETTINGS_PATH = Path(__file__).with_name("settings.yaml")


def _configured_function() -> str:
	with SETTINGS_PATH.open(encoding="utf-8") as settings_file:
		settings = yaml.safe_load(settings_file) or {}
	return settings.get("dividend_witholding_func", "us_investor")


def _us_investor(params: dict[str, Any]) -> tuple[float, float, float, float]:
	gross_dividend = float(params.get("dividend_amount", 0.0))
	if gross_dividend <= 0:
		return gross_dividend, gross_dividend, gross_dividend, 0.0

	is_qualified = bool(params.get("is_qualified", True))
	is_foreign = bool(params.get("is_foreign", False))
	foreign_rate = float(params.get("foreign_withholding_rate", 0.0))
	profile: TaxProfile = params["tax_profile"]
	foreign_tax_withheld = gross_dividend * foreign_rate if is_foreign else 0.0
	fed_rate = 0.15 if is_qualified else 0.24
	us_tax = (
		gross_dividend * (fed_rate + profile.state_tax_rate)
		+ calculate_niit_tax(profile, gross_dividend)
	)
	if is_foreign and params.get("claim_foreign_tax_credit", True):
		us_tax = max(0.0, us_tax - foreign_tax_withheld)

	total_tax = foreign_tax_withheld + us_tax
	return gross_dividend, gross_dividend - total_tax, gross_dividend, total_tax


def calculate_dividend_tax(
	params: dict[str, Any],
) -> tuple[float, float, float, float]:
	"""Return dividend-adjusted value as (original, post-tax, original, tax)."""
	functions = {"us_investor": _us_investor}
	function_name = _configured_function()
	try:
		function = functions[function_name]
	except KeyError as error:
		raise ValueError(f"Unknown dividend withholding function: {function_name}") from error
	return function(params)
