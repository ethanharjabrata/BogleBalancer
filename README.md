# BogleBalancer
A test to see if Reinforcement Learning techniques can be used to assist in portfolio rebalancing for Boglehead investors.

## Overview
BogleBalancer explores whether modern Deep Reinforcement Learning (DRL) can enhance classic passive portfolio management (e.g., Boglehead Three-Fund strategies) without abandoning low-cost, long-term index investing principles. While conventional Boglehead rebalancing relies on static calendar schedules or rigid percentage thresholds, this project evaluates if an RL agent can learn intelligent, friction-aware execution across an investor's time horizon.

## Research Objectives
This project evaluates the viability of DRL agents in portfolio management by testing five primary hypotheses:

1. **Optimal Glidepath Discovery:** Can the agent autonomously determine optimal portfolio weights and dynamically shift asset allocation relative to time remaining until retirement?
2. **Policy Stability & Convergence:** Does the agent converge on a consistent, stable policy regardless of training data bootstrap techniques, historical data sampling, or seed initialization?
3. **Black Swan Resilience:** Does the policy resist catastrophic collapse during extreme tail-risk events? *(Note: While stress-tested against severe historical and synthetic market shocks, the framework acknowledges that true black swan risks remain inherently unquantifiable).*
4. **Friction & Tax Awareness:** Can the reward structure force the model to account for real-world market drag, including brokerage fees, bid-ask spreads, and short/long-term capital gains tax implications across account types?
5. **Tactical Contrarian Tilt:** Can the agent leverage macro bull/bear indicators to make disciplined, minor contrarian rebalancing adjustments (buying deep dips, taking minor profits in hyper-extended regimes) without drifting into active trading?

## Core System Architecture

* **State Space ($S_t$):**
  * Current portfolio weights ($w_i$)
  * Time-to-horizon / investor age progress
  * Accumulated unrealized capital gains/losses per asset class
  * Market regime signals (e.g., volatility metrics, valuation indicators, momentum context)
* **Action Space ($A_t$):** Continuous weight adjustments or discrete rebalancing trade signals across target index funds.
* **Reward Function ($R_t$):** Net utility maximization prioritizing risk-adjusted returns (e.g., Sharpe/Sortino ratios) penalized heavily for transaction fees, tax drag, and severe downside drawdowns.

## Project Roadmap

* [ ] **Phase 1: Baseline Benchmark** — Establish baseline performance metrics against classic static Boglehead strategies (e.g., 80/20 fixed, target-date glidepaths).

## Disclaimer
*This repository is strictly for academic research and experimental simulation. It does not constitute financial advice. Machine learning models trained on historical financial data can overfit, fail on out-of-distribution events, and generate unintended trading behaviors.*