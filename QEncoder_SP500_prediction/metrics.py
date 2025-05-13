import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Mean Absolute Percentage Error (MAPE)
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Calculates maximum drawdown — the largest drop from a peak to a trough
def max_drawdown(returns):
    cumulative = np.cumprod(1 + returns)  # Cumulative return over time
    peak = np.maximum.accumulate(cumulative)  # Running max value seen so far
    drawdown = (cumulative - peak) / peak  # Drop from peak
    return np.min(drawdown)  # Max drawdown (worst drop)


# Sharpe Ratio = (Average excess return) / (Standard deviation of return)
def sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)


# Sortino Ratio = (Average excess return) / (Standard deviation of downside returns)
def sortino_ratio(returns, risk_free_rate=0.0):
    negative_returns = returns[returns < risk_free_rate]  # Only negative returns
    downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 1e-8
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / (downside_std + 1e-8)


# Main evaluation function — prints all metrics
def metrics(y_pred, y_true):
    y_pred = np.array(y_pred).flatten()  # Convert model output to 1D
    y_true = np.array(y_true).flatten()

    # Standard metrics
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape_val = mape(y_true, y_pred)

    # Financial metrics
    returns = np.diff(y_pred) / y_pred[:-1]  # Daily returns
    ret_pct = np.sum(returns) * 100  # Total return %
    sharpe = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)
    max_dd = max_drawdown(returns)

    # Print all metrics
    print(f"R2 Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape_val:.2f}%")
    print(f"Return %: {ret_pct:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Sortino Ratio: {sortino:.4f}")
    print(f"Max Drawdown: {max_dd:.4f}")
