import numpy as np
from typing import Callable

def approx_gradient(func: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Approximates the gradient of a scalar-valued function using central finite differences.
    
    Note: A loop is necessary here because the black-box function `func` is not 
    guaranteed to support batched/2D array broadcasting.

    Parameters:
        func (callable): A scalar-valued function taking a 1D np.ndarray.
        x (np.ndarray): The evaluation point. Shape (n_dims,).
        h (float): The step size for finite differences.

    Returns:
        np.ndarray: The gradient vector. Shape (n_dims,).
    
    Complexity:
        Time: O(n_dims * T_func) where T_func is the time to evaluate func.
        Space: O(n_dims) to store the gradient.
    """
    if x.ndim != 1:
        raise ValueError(f"Input x must be 1D, got shape {x.ndim}")

    n_dims = x.shape[0]
    grad = np.zeros(n_dims, dtype=float)
    
    # Identity matrix scaled by step size
    H = np.eye(n_dims) * h
    
    for i in range(n_dims):
        # Central difference: (f(x + h) - f(x - h)) / 2h
        val_plus = func(x + H[i])
        val_minus = func(x - H[i])
        grad[i] = (val_plus - val_minus) / (2.0 * h)
        
    return grad

def approx_jacobian(func: Callable[[np.ndarray], np.ndarray], x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Approximates the Jacobian matrix of a vector-valued function using central finite differences.

    Parameters:
        func (callable): A function taking a 1D array of shape (n,) and returning a 1D array of shape (m,).
        x (np.ndarray): The evaluation point. Shape (n,).
        h (float): The step size for finite differences.

    Returns:
        np.ndarray: The Jacobian matrix of shape (m, n).

    Raises:
        ValueError: If input is not 1D or the function output is not 1D.

    Complexity:
        Time: O(n * T_func).
        Space: O(m * n) to store the Jacobian.
    """
    if x.ndim != 1:
        raise ValueError(f"Input x must be 1D, got shape {x.ndim}")

    # Evaluate once to get output dimension (m)
    f_x = func(x)
    
    if not isinstance(f_x, np.ndarray) or f_x.ndim != 1:
        raise ValueError("Function must return a 1D numpy array.")
        
    n = x.shape[0]
    m = f_x.shape[0]
    
    jacobian = np.zeros((m, n), dtype=float)
    H = np.eye(n) * h
    
    for i in range(n):
        val_plus = func(x + H[i])
        val_minus = func(x - H[i])
        # Vectorized over the m outputs
        jacobian[:, i] = (val_plus - val_minus) / (2.0 * h)
        
    return jacobian