"""Run with: .venv\\Scripts\\python -m examples.baseline"""

from strategies.baseline import simulate_baseline
from utils.summary_statistics.calculate_summary import compute_portfolio_metrics


def main() -> None:
    gross, net = simulate_baseline()
    if gross.empty or net.empty or gross.iloc[-1] < net.iloc[-1]:
        raise RuntimeError("Gross ending value must be greater than or equal to net ending value")
    for label, history in (("Gross / no drag", gross), ("Net / drag-adjusted", net)):
        print(f"\n{label}")
        for metric, value in compute_portfolio_metrics(history).items():
            print(f"{metric}: {value}")
        print(f"Ending value: {history.iloc[-1]:,.2f}")


if __name__ == "__main__":
    main()
