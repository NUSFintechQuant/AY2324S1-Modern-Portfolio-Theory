import cvxpy as cp
import numpy as np
import pandas as pd
import math

def get_mean_and_variance(assets):
    df = pd.read_csv("../data/{}.csv".format(assets[0]))
    date_series = df["datadate"]
    price = df["prccd"]
    df = pd.DataFrame({"date" : date_series, assets[0] : price})
    for i in range(1, len(assets)):
        df1 = pd.read_csv("../data/{}.csv".format(assets[i]))
        date_series1 = df1["datadate"]
        price1 = df1["prccd"]
        ret_df1 = pd.DataFrame({"date" : date_series1, assets[i] : price1})
        df = pd.merge(df, ret_df1, how = "inner", on = "date")

    df['date'] = pd.to_datetime(df['date'])

    # Filter the DataFrame based on the condition
    df = df[(df['date'] >= '2006-12-11') & (df['date'] <= '2007-01-11')]

    # Apply the function to specific columns using df.apply
    df[assets] = df[assets].apply(np.log).diff();
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
      w = cp.Variable(5)
      lambda_entropy = 0.01
      shannon_entropy = cp.sum(cp.entr(w))
      obj = cp.Minimize(cp.quad_form(w, covariance_matrix))
      const = [
          w[0]<=.3,w[1]<=.3,w[2]<=.3,w[3]<=.3,w[4]<=.3,cp.sum(w) == 1, w >= 0, w.T @ expected_returns == returns_min_annualized
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

def mean_variance_optimization():
    [expected_returns, covariance_matrix] = get_mean_and_variance(['commodities', 'gold', 'us_small_cap', 'snp500', 'us_bonds'])
    [efficient_portfolio_returns, efficient_portfolio_volatilities, sharpe_ratios] = efficient_portfolio_generation(1.00, 0.02, covariance_matrix, expected_returns)
    sharpe_ratios = sharpe_ratios.rename(index={0: 'Returns', 1: 'Risk', 2: 'Sharpe Ratio', 3: 'Weights'})
    # get the weights, returns and risk of portfolio that has the highest sharpe ratio
    market_portfolio = sharpe_ratios[sharpe_ratios.loc['Sharpe Ratio'].astype(float).idxmax()]
    print(market_portfolio.values)
    return market_portfolio

if __name__ == "__main__":
    mean_variance_optimization()
