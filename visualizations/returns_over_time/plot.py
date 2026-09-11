import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from visualizations.returns_over_time.returns import calculate_monthly_returns

def plot_ticker_monthly_returns(
    tickers: list[str], start_year: int, end_year: int
):
  """Collects monthly performance data for multiple tickers using the helper function

  and plots a Seaborn multi-line chart on the same axis.
  """
  records = []
  for data in calculate_monthly_returns(tickers, start_year, end_year):
    records.append({
        "Date": pd.to_datetime(data["buy_date"]),
        "Ticker": data["ticker"],
        "Monthly Return (%)": float(data["monthly_return"].replace("%", "")),
    })

  # Create DataFrame
  df_results = pd.DataFrame(records)

  # Plot configuration
  sns.set_theme(style="whitegrid", palette="deep")
  plt.figure(figsize=(14, 6))

  # Generate multi-line plot
  ax = sns.lineplot(
      data=df_results,
      x="Date",
      y="Monthly Return (%)",
      hue="Ticker",
      marker="o",
      linewidth=2.2,
      alpha=0.85,
  )

  # Graph styling
  plt.title(
      f"Monthly Returns Comparison ({', '.join(tickers)}) [{start_year}–{end_year}]",
      fontsize=14,
      fontweight="bold",
      pad=15,
  )
  plt.xlabel("Date", fontsize=11, labelpad=10)
  plt.ylabel("Monthly Return (%)", fontsize=11, labelpad=10)

  # Add benchmark 0% reference line
  plt.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

  # Legend & Layout
  plt.legend(title="Asset", title_fontsize="10", loc="upper left", frameon=True)
  plt.tight_layout()
  plt.savefig("./visualizations/returns_over_time/monthly_returns_comparison.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
  # Specify assets and time frame
  target_tickers = ["VTI", "VXUS", "BND"]
  plot_ticker_monthly_returns(
      tickers=target_tickers, start_year=2021, end_year=2025
  )