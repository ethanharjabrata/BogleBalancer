"""Two-asset contribution-funded baseline portfolio simulation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from utils.drag_funcs.brokerage_fees import calculate_brokerage_fees
from utils.drag_funcs.capital_gains import TaxProfile, calculate_capital_gains_tax
from utils.drag_funcs.dividend_witholding import calculate_dividend_tax

SETTINGS_PATH = Path(__file__).with_name("settings.yaml")
DRAG_SETTINGS_PATH = Path(__file__).parents[1] / "utils" / "drag_funcs" / "settings.yaml"
_FREQUENCIES = {"daily", "monthly", "quarterly", "yearly"}
_WEIGHT_TOLERANCE = 1e-9

@dataclass(frozen=True)
class _Config:
    start_year: int; end_year: int; tickers: tuple[str, str]; weights: np.ndarray
    rebalance_frequency: str; threshold: float; inflow_frequency: str; inflow_value: float

def load_settings(config_path: Path | str = SETTINGS_PATH) -> dict[str, Any]:
    """Safely read a YAML mapping."""
    try:
        with Path(config_path).open(encoding="utf-8") as stream: result = yaml.safe_load(stream)
    except FileNotFoundError as exc: raise ValueError(f"Baseline settings file does not exist: {config_path}") from exc
    if not isinstance(result, dict): raise ValueError("Baseline settings must be a YAML mapping")
    return result

def _number(value: Any, name: str, non_negative: bool = False) -> float:
    if isinstance(value, bool): raise ValueError(f"{name} must be numeric")
    try: result = float(value)
    except (TypeError, ValueError) as exc: raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(result) or non_negative and result < 0: raise ValueError(f"{name} must be a non-negative finite number" if non_negative else f"{name} must be finite")
    return result

def _frequency(value: Any, name: str) -> str:
    result = value.lower().strip() if isinstance(value, str) else ""
    if result not in _FREQUENCIES: raise ValueError(f"Unsupported {name}: {value!r}; supported values are {sorted(_FREQUENCIES)}")
    return result

def _validate_config(s: dict[str, Any]) -> _Config:
    try: timeframe, baseline, inflows = s["timeframe"], s["baseline"], s["inflows"]
    except KeyError as exc: raise ValueError(f"Missing required baseline settings section: {exc.args[0]}") from exc
    if not all(isinstance(x, dict) for x in (timeframe, baseline, inflows)): raise ValueError("timeframe, baseline, and inflows must be mappings")
    start, end = timeframe.get("start_year"), timeframe.get("end_year")
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int) or start > end: raise ValueError("timeframe start_year and end_year must be integers with start_year <= end_year")
    tickers = (baseline.get("us_equities_ticker"), baseline.get("us_bonds_ticker"))
    if not all(isinstance(x, str) and x.strip() for x in tickers): raise ValueError("baseline requires two non-empty ticker strings")
    weights = np.array([_number(baseline.get("us_equities_weight"), "equity weight"), _number(baseline.get("us_bonds_weight"), "bond weight")])
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1, atol=_WEIGHT_TOLERANCE, rtol=0): raise ValueError(f"Baseline weights must be non-negative and sum to 1.0 within {_WEIGHT_TOLERANCE}")
    return _Config(start, end, (tickers[0].strip(), tickers[1].strip()), weights, _frequency(baseline.get("rebalance_frequency"), "rebalance frequency"), _number(baseline.get("rebalance_threshold"), "rebalance threshold", True), _frequency(inflows.get("frequency"), "inflow frequency"), _number(inflows.get("value"), "inflow value", True))

def _tax_profile(path: Path | str) -> TaxProfile:
    s = load_settings(path).get("us_investor_settings")
    if not isinstance(s, dict): raise ValueError("drag settings require a us_investor_settings mapping")
    if s.get("filing_status") not in {"single", "mfj"}: raise ValueError("us_investor_settings.filing_status must be 'single' or 'mfj'")
    return TaxProfile(s["filing_status"], _number(s.get("ordinary_taxable_income"), "ordinary taxable income", True), _number(s.get("magi"), "MAGI", True), _number(s.get("state_tax_rate"), "state tax rate", True))

def _clean_prices(prices: pd.DataFrame, tickers: tuple[str, str]) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame) or prices.empty: raise ValueError("Market prices must be a non-empty DataFrame")
    r = prices.copy().reindex(columns=list(tickers)); r.index = pd.to_datetime(r.index).tz_localize(None); r = r.sort_index()
    if r.index.has_duplicates or r.isna().any().any() or not np.isfinite(r.to_numpy(float)).all() or (r <= 0).any().any(): raise ValueError("Market prices must have unique dates and complete, positive finite closes")
    return r.astype(float)

def _clean_dividends(dividends: pd.DataFrame | None, tickers: tuple[str, str]) -> pd.DataFrame:
    if dividends is None: return pd.DataFrame(0., index=pd.DatetimeIndex([]), columns=list(tickers))
    if not isinstance(dividends, pd.DataFrame): raise ValueError("Dividends must be a DataFrame")
    r = dividends.copy().reindex(columns=list(tickers)).fillna(0.); r.index = pd.to_datetime(r.index).tz_localize(None)
    if not np.isfinite(r.to_numpy(float)).all() or (r < 0).any().any(): raise ValueError("Dividends must be non-negative finite values")
    return r.astype(float)

def load_market_data(tickers: tuple[str, str], start_year: int, end_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download both tickers once, using raw closes with separately reinvested dividends.

    Yahoo adjusted closes embed distributions.  Raw closes are deliberately used here
    so that explicitly modelled gross and post-tax dividend reinvestment is not
    double counted. Missing closes are rejected, never filled.
    """
    import yfinance as yf
    raw = yf.download(list(tickers), start=f"{start_year}-01-01", end=f"{end_year + 1}-01-01", auto_adjust=False, actions=True, progress=False)
    if raw.empty: raise ValueError("Market-data provider returned no data")
    def get(name: str) -> pd.DataFrame:
        if name not in raw: raise ValueError(f"Market data has no {name!r} field")
        x = raw[name]; return (x.to_frame(tickers[0]) if isinstance(x, pd.Series) else x).reindex(columns=list(tickers))
    return _clean_prices(get("Close"), tickers), _clean_dividends(get("Dividends"), tickers)

def eligible_dates(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """First available trading date per calendar period (or every date for daily)."""
    frequency = _frequency(frequency, "frequency")
    if frequency == "daily": return pd.DatetimeIndex(index)
    keys = index.to_period({"monthly":"M", "quarterly":"Q", "yearly":"Y"}[frequency])
    return pd.DatetimeIndex(index[~keys.duplicated()])

def _buy(shares, cash, prices, proportions, lots, date, net):
    for i, amount in enumerate(cash * proportions):
        if amount <= 0: continue
        fee = calculate_brokerage_fees({"value": amount, "transaction_type":"purchase"})[3] if net else 0.
        amount = min(amount, cash); qty = max(0., (amount - min(fee, amount)) / prices[i]); shares[i] += qty; cash -= amount
        if lots is not None and qty: lots[i].append({"shares":qty, "basis":qty * prices[i], "date":date})
    return max(cash, 0.)

def _sell(shares, cash, prices, targets, lots, date, profile):
    for i, excess in enumerate(np.maximum(shares * prices - targets, 0.)):
        qty = min(shares[i], excess / prices[i]); gross = qty * prices[i]
        if qty <= 0: continue
        fee = calculate_brokerage_fees({"value":gross, "transaction_type":"sale"})[3] if profile else 0.; tax = 0.
        if lots is not None:
            left = qty
            while left > 1e-12 and lots[i]:
                lot = lots[i][0]; used = min(left, lot["shares"]); basis = lot["basis"] * used / lot["shares"]; gain = used * prices[i] - basis
                if gain > 0: tax += calculate_capital_gains_tax({"gain":gain, "holding_period_days":(date-lot["date"]).days, "tax_profile":profile})[3]
                lot["shares"] -= used; lot["basis"] -= basis; left -= used
                if lot["shares"] <= 1e-12: lots[i].pop(0)
        shares[i] -= qty; cash += max(0., gross-fee-tax)
    return cash

def _run(prices: pd.DataFrame, dividends: pd.DataFrame, cfg: _Config, profile: TaxProfile | None) -> pd.Series:
    dates = prices.index; rebalances, inflows = set(eligible_dates(dates, cfg.rebalance_frequency)), set(eligible_dates(dates, cfg.inflow_frequency)); shares = np.zeros(2); cash = 0.; lots = [[], []] if profile else None; history=[]
    for date, row in prices.iterrows():
        price = row.to_numpy(float)
        if date in dividends.index:
            for i, dividend in enumerate(dividends.loc[date].to_numpy(float)):
                gross = shares[i]*dividend
                if gross > 0:
                    post = calculate_dividend_tax({"dividend_amount":gross, "is_qualified":True, "is_foreign":False, "foreign_withholding_rate":0., "tax_profile":profile})[1] if profile else gross
                    qty=post/price[i]; shares[i]+=qty
                    if lots is not None: lots[i].append({"shares":qty,"basis":post,"date":date})
        if date in inflows: cash = _buy(shares, cash+cfg.inflow_value, price, cfg.weights, lots, date, profile is not None)
        total = cash + float(shares @ price); actual = shares*price/total if total else cfg.weights
        if date in rebalances and np.max(np.abs(actual-cfg.weights)) >= cfg.threshold:
            targets=total*cfg.weights; cash=_sell(shares,cash,price,targets,lots,date,profile); deficits=np.maximum(targets-shares*price,0.)
            if cash > 0 and deficits.sum() > 0: cash=_buy(shares,cash,price,deficits/deficits.sum(),lots,date,profile is not None)
        history.append(cash+float(shares@price))
    return pd.Series(history,index=dates,dtype=float,name="net_portfolio_value" if profile else "gross_portfolio_value")

def simulate_baseline(config_path: Path | str = SETTINGS_PATH, drag_config_path: Path | str = DRAG_SETTINGS_PATH, *, market_data: pd.DataFrame | None = None, dividends: pd.DataFrame | None = None) -> tuple[pd.Series, pd.Series]:
    """Return independent float-valued gross/net histories indexed by trading date."""
    cfg = _validate_config(load_settings(config_path)); profile = _tax_profile(drag_config_path)
    prices, events = load_market_data(cfg.tickers,cfg.start_year,cfg.end_year) if market_data is None else (_clean_prices(market_data,cfg.tickers),_clean_dividends(dividends,cfg.tickers))
    gross, net = _run(prices,events,cfg,None), _run(prices,events,cfg,profile)
    if gross.empty or net.empty or gross.isna().any() or net.isna().any(): raise ValueError("Simulation produced an invalid empty or missing history")
    return gross, net
