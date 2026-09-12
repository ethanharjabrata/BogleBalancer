from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SETTINGS_PATH = Path(__file__).with_name("settings.yaml")
LTCG_BRACKETS_2026 = {
	"single": [(49_450, 0.00), (545_500, 0.15), (float("inf"), 0.20)],
	"mfj": [(98_900, 0.00), (613_700, 0.15), (float("inf"), 0.20)],
}
NIIT_THRESHOLDS = {"single": 200_000, "mfj": 250_000}


@dataclass(frozen=True)
class TaxProfile:
	filing_status: str
	ordinary_taxable_income: float
	magi: float
	state_tax_rate: float


def _configured_function() -> str:
	with SETTINGS_PATH.open(encoding="utf-8") as settings_file:
		settings = yaml.safe_load(settings_file) or {}
	return settings.get("capital_gains_func", "us_investor")


def calculate_niit_tax(profile: TaxProfile, net_investment_income: float) -> float:
	threshold = NIIT_THRESHOLDS.get(profile.filing_status, 200_000)
	excess_income = max(0.0, profile.magi - threshold)
	taxable_niit_base = min(max(0.0, net_investment_income), excess_income)
	return taxable_niit_base * 0.038


def _us_investor(params: dict[str, Any]) -> tuple[float, float, float, float]:
	gain = float(params.get("gain", 0.0))
	if gain <= 0:
		return gain, gain, gain, 0.0

	profile: TaxProfile = params["tax_profile"]
	holding_days = int(params.get("holding_period_days", 0))
	if holding_days > 365:
		brackets = LTCG_BRACKETS_2026.get(
			profile.filing_status, LTCG_BRACKETS_2026["single"]
		)
		fed_rate = next(
			rate
			for limit, rate in brackets
			if profile.ordinary_taxable_income < limit
		)
	else:
		fed_rate = 0.24

	niit_tax = calculate_niit_tax(profile, gain)
	total_tax = gain * (fed_rate + profile.state_tax_rate) + niit_tax
	post_tax_value = gain - total_tax
	return gain, post_tax_value, gain, total_tax


def calculate_capital_gains_tax(
	params: dict[str, Any],
) -> tuple[float, float, float, float]:
	"""Return capital-gains-adjusted value as (original, post-tax, original, tax)."""
	functions = {"us_investor": _us_investor}
	function_name = _configured_function()
	try:
		function = functions[function_name]
	except KeyError as error:
		raise ValueError(f"Unknown capital gains function: {function_name}") from error
	return function(params)
