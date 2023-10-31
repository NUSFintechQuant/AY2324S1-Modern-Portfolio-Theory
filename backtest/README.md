# Internal Backtesting

## Features

- Multiple Input Types
    - [ ] Starting portfolio weights

- Intermediate Data
    - [ ] Daily returns

- Outputs
    - [ ] Sharpe
    - [ ] Max drawdown
    - [ ] PnL
    - [ ] Beta (relative to SnP500)

### Historical Data

- Should be flexible where user can provide own historical returns to backtest.
- If user does not provide data, backtester either:
    1. scrapes data from somewhere (make use of scrapers built);
        - could be yfinance. can also look into WRDS.
    2. taps into some cold database we have (pre-processed & cleaned data)
        - better option but not immediately necessary

### Option for Event-Driven

We eventually want to simulate real market trading conditions which involves *dynamic portfolio updates*. K.I.V.
