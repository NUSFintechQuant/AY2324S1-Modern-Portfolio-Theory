import numpy as np

def compute_drawdown(data: np.array) -> float:
    raise NotImplementedError("Function not yet implemented.")

def compute_var(data: np.ndarray, confidence_level: float) -> float:
    """
    Compute the Value at Risk (VaR) for a given n-dimensional numpy array representing investment returns.
    
    Parameters:
    -----------
    data : np.ndarray
        An n-dimensional numpy array representing the time series data of investment returns.
    confidence_level : float
        The desired confidence level for the VaR calculation, e.g., 0.95 for 95% confidence.
        
    Returns:
    --------
    float
        The calculated VaR for the provided data at the given confidence level.
        
    Example:
    --------
    >>> returns = np.array([-0.02, 0.03, -0.01, 0.04])
    >>> compute_var(returns, 0.95)
    # Your example output here, e.g., -0.03
    
    Notes:
    ------
    Value at Risk (VaR) quantifies the maximum potential loss an investment portfolio could face.
    """
    raise NotImplementedError("Function not yet implemented.")

def compute_cvar(data: np.ndarray, confidence_level: float) -> float:
    raise NotImplementedError("Function not yet implemented.")
