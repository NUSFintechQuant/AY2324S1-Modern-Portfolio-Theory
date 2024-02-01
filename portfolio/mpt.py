import cvxpy as cp
import numpy as np
import pandas as pd
import math

def get_mean_and_variance():
    df = pd.read_csv("../data/{}.csv".format("snp500"))
    date_series = df["datadate"]
    price = df["prccd"]
    ret_df = pd.DataFrame({"date" : date_series, "snp500" : price})

    df1 = pd.read_csv("../data/{}.csv".format("us_bonds"))
    date_series1 = df1["datadate"]
    price1 = df1["prccd"]
    ret_df1 = pd.DataFrame({"date" : date_series1, "us_bonds" : price1})

    df = pd.merge(ret_df, ret_df1, how = "inner", on = "date")
    df['date'] = pd.to_datetime(df['date'])

    # Filter the DataFrame based on the condition
    df = df[(df['date'] >= '2006-12-11') & (df['date'] <= '2007-01-11')]
    columns_to_apply = ['snp500', 'us_bonds']

    # Apply the function to specific columns using df.apply
    df[columns_to_apply] = df[columns_to_apply].apply(np.log).diff();
    expected_returns = df.mean() * 252
    covariance_matrix = df.cov().to_numpy()
    return [expected_returns, covariance_matrix]

def efficient_portfolio_generation(max_return, increment, covariance_matrix, expected_returns):
    returns_min_annualized = -1.00
    num_iterations = (int) ((max_return - returns_min_annualized) / increment)
    efficient_portfolio_returns = np.empty((num_iterations,))
    efficient_portfolio_volatilities = np.empty((num_iterations,))
    max_index = num_iterations
    sharpe_ratios = pd.DataFrame()
    index = 0
    for iter in range(num_iterations):
      w = cp.Variable(2)

      obj = cp.Minimize(w.T @ covariance_matrix @ w)
      const = [
          cp.sum(w) == 1, w >= 0, w.T @ expected_returns == returns_min_annualized
      ]
      prob = cp.Problem(obj, const)
      opt_v = prob.solve()
      if (math.isinf(opt_v)):
        max_index -= 1
        efficient_portfolio_returns.resize((max_index,))
        efficient_portfolio_volatilities.resize((max_index,))
      else:
        risk_opt = (opt_v * 252) ** 0.5
        w_opt = w.value
        efficient_portfolio_returns[index] = returns_min_annualized
        efficient_portfolio_volatilities[index] = risk_opt
        sharpe_ratio = (returns_min_annualized - 0.03) / risk_opt
        sharpe_ratios[iter] = [returns_min_annualized, risk_opt, sharpe_ratio, w_opt]
        index += 1
      returns_min_annualized += increment
    return [efficient_portfolio_returns, efficient_portfolio_volatilities, sharpe_ratios]

def mean_variance_optimization(covariance_matrix, expected_returns):
    [expected_returns, covariance_matrix] = get_mean_and_variance(['snp500', 'us_bonds'])
    [efficient_portfolio_returns, efficient_portfolio_volatilities, sharpe_ratios] = efficient_portfolio_generation(1.00, 0.02, covariance_matrix, expected_returns)
    sharpe_ratios = sharpe_ratios.rename(index={0: 'Returns', 1: 'Risk', 2: 'Sharpe Ratio', 3: 'Weights'})
    # get the weights, returns and risk of portfolio that has the highest sharpe ratio
    market_portfolio = sharpe_ratios[sharpe_ratios.loc['Sharpe Ratio'].astype(float).idxmax()]
    return market_portfolio
