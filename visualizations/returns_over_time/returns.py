import pandas as pd
import yfinance as yf


def _monthly_first_trading_days(price_data: pd.DataFrame) -> pd.DataFrame:
    """Return the first available close and date for every calendar month."""
    prices = price_data[["Close"]].rename_axis("date").reset_index()
    prices["month"] = prices["date"].dt.tz_localize(None).dt.to_period("M")
    return (
        prices.groupby("month", sort=True, as_index=False)
        .first()
        .rename(columns={"date": "trading_date", "Close": "close"})
    )


def _build_return_record(
    ticker_symbol: str, buy_row: pd.Series, sell_row: pd.Series
) -> dict:
    buy_date = buy_row["trading_date"]
    sell_date = sell_row["trading_date"]
    buy_price = buy_row["close"]
    sell_price = sell_row["close"]
    holding_days = (sell_date - buy_date).days
    total_return = (sell_price - buy_price) / buy_price
    annualized_return = ((1 + total_return) ** (365.0 / holding_days)) - 1

    return {
        "ticker": ticker_symbol,
        "buy_date": buy_date.strftime("%Y-%m-%d"),
        "buy_price": round(buy_price, 2),
        "sell_date": sell_date.strftime("%Y-%m-%d"),
        "sell_price": round(sell_price, 2),
        "holding_days": holding_days,
        "monthly_return": f"{total_return * 100:.2f}%",
        "annualized_return": f"{annualized_return * 100:.2f}%",
    }


def _load_monthly_prices(
    ticker_symbol: str, start_year: int, end_year: int
) -> pd.DataFrame:
    start_date = pd.Timestamp(year=start_year, month=1, day=1)
    end_date = pd.Timestamp(year=end_year + 1, month=2, day=1)
    ticker = yf.Ticker(ticker_symbol)
    price_data = ticker.history(
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=True,
    )
    if price_data.empty:
        raise ValueError(f"No price data found for {ticker_symbol}.")
    return _monthly_first_trading_days(price_data)


def calculate_monthly_returns(
    ticker_symbols: list[str], start_year: int, end_year: int
) -> list[dict]:
    """Calculate all one-month returns with one history request per ticker."""
    requested_months = pd.period_range(
        f"{start_year}-01", f"{end_year}-12", freq="M"
    )
    records = []

    for ticker_symbol in ticker_symbols:
        monthly_prices = _load_monthly_prices(
            ticker_symbol, start_year, end_year
        ).set_index("month")
        buy_prices = monthly_prices.reindex(requested_months)
        sell_prices = monthly_prices.reindex(requested_months + 1)
        sell_prices.index = requested_months
        valid_rows = buy_prices["close"].notna() & sell_prices["close"].notna()

        for month in requested_months[valid_rows]:
            records.append(
                _build_return_record(
                    ticker_symbol,
                    buy_prices.loc[month],
                    sell_prices.loc[month],
                )
            )

    return records


def calculate_one_month_annualized_return(
    ticker_symbol: str, buy_year: int, buy_month: int
) -> dict:
    """Calculates the annualized return for holding an asset for 1 month starting on the 1st.

    Args:
        ticker_symbol: Ticker symbol (e.g., 'VTI', 'VXUS', 'BND')
        buy_year: Year of purchase (e.g., 2023)
        buy_month: Month of purchase (1-12)

    Returns:
        Dictionary containing transaction details and performance metrics.
    """
    monthly_prices = _load_monthly_prices(ticker_symbol, buy_year, buy_year)
    buy_month_period = pd.Period(f"{buy_year}-{buy_month:02d}", freq="M")
    prices_by_month = monthly_prices.set_index("month")

    if buy_month_period not in prices_by_month.index:
        raise ValueError(f"No trading days found in buy month: {buy_month_period}.")
    if buy_month_period + 1 not in prices_by_month.index:
        raise ValueError(
            f"No trading days found in sell month: {buy_month_period + 1}."
        )

    return _build_return_record(
        ticker_symbol,
        prices_by_month.loc[buy_month_period],
        prices_by_month.loc[buy_month_period + 1],
    )


if __name__ == "__main__":
    # Example usage
    ticker = "VTI"
    year = 2024
    month = 1  # January

    result = calculate_one_month_annualized_return(ticker, year, month)

    print("--- Portfolio Single-Month Holding Performance ---")
    for key, value in result.items():
        print(f"{key.replace('_', ' ').title()}: {value}")