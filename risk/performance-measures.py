import numpy as np

def compute_alpha(data: np.ndarray) -> float:
    """
    Compute the alpha value for a given n-dimensional numpy array representing investment returns.
    
    Parameters:
    -----------
    data : np.ndarray
        An n-dimensional numpy array representing the time series data of investment returns.
        
    Returns:
    --------
    float
        The calculated alpha of the provided data.
        
    Example:
    --------
    >>> returns = np.array([0.05, 0.02, -0.01, 0.03])
    >>> compute_alpha(returns)
    # Your example output here, e.g., 0.0125
    
    Notes:
    ------
    Alpha is used in finance to represent the abnormal rate of return on a security or portfolio 
    in comparison to the expected return given its beta and the expected market returns.
    A positive alpha suggests the investment has performed better than its beta would predict.
    """
    raise NotImplementedError("Function not yet implemented.")

def compute_sharpe(data: np.ndarray) -> float:
    raise NotImplementedError("Function not yet implemented.")

def compute_sortino(data: np.ndarray) -> float:
    raise NotImplementedError("Function not yet implemented.")

