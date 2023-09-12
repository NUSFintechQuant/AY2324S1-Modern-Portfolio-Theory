import numpy as np

def compute_variance(data: np.ndarray) -> float:
    """
    Compute the variance of a given n-dimensional numpy array.
    
    Parameters:
    -----------
    data : np.ndarray
        An n-dimensional numpy array representing the time series data.
        
    Returns:
    --------
    float
        The variance of the provided data.
    
    note: we might want to provide some other measures later down the line like the variance of the metric calculated via some method like bootstrapping and random sampling.
        
    Example:
    --------
    >>> data = np.array([2.5, 3.1, 4.2, 5.8])
    >>> compute_variance(data)
    1.8158333333333332
    
    Notes:
    ------
    Variance measures how far a set of numbers are spread out from their average value.
    """
    raise NotImplementedError("Function not yet implemented.")

def compute_std_dev(data: np.ndarray) -> float:
    raise NotImplementedError("Function not yet implemented.")

def compute_expected_return(data: np.ndarray) -> float:
    raise NotImplementedError("Function not yet implemented.")
