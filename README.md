# NVDA-Time-Series

A Python time-series project that downloads historical NVIDIA (NVDA) stock price data, trains a simple regression model using lagged features, and generates short-term price forecasts.

## Features
- Downloads NVDA daily historical prices (Yahoo Finance via `yfinance`)
- Creates lag-based features (past N days) for time-series modeling
- Trains a Linear Regression model and evaluates it with MAE
- Predicts the next few days using a rolling forecast
- Saves a moving-average chart to `reports/figures/`