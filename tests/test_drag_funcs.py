from concurrent.futures import ThreadPoolExecutor

import pytest

from utils.drag_funcs.brokerage_fees import calculate_brokerage_fees
from utils.drag_funcs.capital_gains import TaxProfile, calculate_capital_gains_tax
from utils.drag_funcs.dividend_witholding import calculate_dividend_tax

PROFILE = TaxProfile(
    filing_status="single",
    ordinary_taxable_income=40_000,
    magi=40_000,
    state_tax_rate=0.05,
)


def test_us_investor_brokerage_has_no_fee():
    assert calculate_brokerage_fees({"value": 1_000.0}) == (1_000.0, 1_000.0, 1_000.0, 0.0)


def test_capital_gains_tax_handles_short_term_and_niit():
    result = calculate_capital_gains_tax(
        {"gain": 1_000.0, "holding_period_days": 30, "tax_profile": PROFILE}
    )
    assert result == pytest.approx((1_000.0, 710.0, 1_000.0, 290.0))


def test_dividend_tax_applies_foreign_withholding_and_credit():
    result = calculate_dividend_tax(
        {
            "dividend_amount": 1_000.0,
            "is_qualified": True,
            "is_foreign": True,
            "foreign_withholding_rate": 0.15,
            "tax_profile": PROFILE,
        }
    )
    assert result == pytest.approx((1_000.0, 800.0, 1_000.0, 200.0))


def test_functions_are_safe_to_call_in_parallel():
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(
            executor.map(
                calculate_brokerage_fees,
                ({"value": value} for value in (1.0, 2.0, 3.0)),
            )
        )
    assert results == [
        (1.0, 1.0, 1.0, 0.0),
        (2.0, 2.0, 2.0, 0.0),
        (3.0, 3.0, 3.0, 0.0),
    ]