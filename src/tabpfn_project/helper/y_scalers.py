'''
This module includes the transformation functions for scaling the target variable (y).
'''

import numpy as np


def max_scaling(y_train, *arrays):
    """
    Min-max normalize arrays by the maximum value in y_train.
    
    Scales y_train and all provided arrays by 1/max(y_train).
    
    Parameters
    ----------
    y_train : np.ndarray
        Training targets used to compute the scaling factor.
    *arrays : np.ndarray
        Additional arrays to apply the same scaling to.
    
    Returns
    -------
    tuple
        Scaled y_train, scaled additional arrays, and the scale factor (1/max).
    """
    y_max = np.max(y_train)
    scale = 1.0 if y_max == 0 else (1.0 / y_max)

    y_train_scaled = y_train * scale

    # Apply scale to all other arrays
    processed_arrays = [arr * scale for arr in arrays]

    return y_train_scaled, *processed_arrays, scale


def log1p_scaling(y_train, *arrays):
    """
    Apply log1p transformation to arrays.
    
    Computes log(1 + y) for y_train and all provided arrays.
    
    Parameters
    ----------
    y_train : np.ndarray
        Training targets to transform.
    *arrays : np.ndarray
        Additional arrays to apply the same transformation to.
    
    Returns
    -------
    tuple
        Transformed y_train and transformed additional arrays.
    """
    y_train_logged = np.log1p(y_train)

    # Apply log to all other arrays
    processed_arrays = [np.log1p(arr) for arr in arrays]

    return y_train_logged, *processed_arrays
