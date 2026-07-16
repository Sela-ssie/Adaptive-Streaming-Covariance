# Adaptive Streaming Covariance Estimation Under Regime Change

This project studies how covariance estimators trade off statistical accuracy,
adaptation speed, numerical stability, portfolio stability, latency, and memory
usage when the underlying covariance structure changes over time.

## Downstream use case

The eventual application is a streaming portfolio-risk engine that:

1. receives a new vector of asset returns;
2. updates a covariance estimate incrementally;
3. determines whether estimated market risk has changed materially;
4. optionally recomputes a risk-controlled portfolio;
5. avoids unnecessary portfolio rebalancing;
6. reports estimated risk, realized out-of-sample risk, turnover, and latency.

## Research question

Under a fixed computational budget, which covariance estimator provides the best
tradeoff among:

- covariance estimation accuracy;
- adaptation following regime changes;
- numerical conditioning;
- portfolio weight stability;
- portfolio turnover;
- update latency;
- memory usage?

## Planned baseline estimators

The baseline study will include:

1. expanding-window sample covariance;
2. rolling-window sample covariance;
3. exponentially weighted covariance;
4. batch Ledoit-Wolf shrinkage;
5. periodically refreshed or streaming shrinkage;
6. a multi-timescale or adaptive estimator.

Advanced methods will not be added until the baseline estimators are validated.

## Current status

The current milestone establishes:

- a transparent batch sample-covariance implementation;
- trusted numerical comparisons against NumPy;
- unit-testing conventions;
- package structure for later estimators.

## Research standards

Results will be reported with clearly defined metrics, repeated simulation trials,
uncertainty estimates, limitations, and negative findings where applicable.

The project does not assume that the most accurate covariance estimator will
necessarily produce the most stable or useful portfolio.