'''
This module contains helper functions for the TabPFN model.
'''

from typing import Any
import torch
import numpy as np
from tabpfn_project.helper.utils import TargetScale, dict_to_cpu
from tabpfn_project.helper.y_scalers import log1p_scaling, max_scaling


def batch_predict_tabpfn(model: Any, X_test: np.ndarray, validation_batch_size: int):
    """
    Batch predictor for TabPFN model.

    Args:
        model: The TabPFN model instance.
        X_test: Input features for testing, shape (B, D).
        validation_batch_size: Number of instances to process per batch.

    Returns:
        A list of dictionaries containing model predictions moved to CPU.
    """
    n_instances = X_test.shape[0]
    tabpfn_preds = []
    with torch.inference_mode():
        for start in range(0, n_instances, validation_batch_size):
            X_batch = X_test[start : start + validation_batch_size]

            # Generate predictions with full distribution output
            preds = model.predict(X_batch, output_type="full")

            # Move tensors to CPU immediately to prevent GPU memory accumulation
            tabpfn_preds.append(dict_to_cpu(preds))
    return tabpfn_preds


def oracle_predict_tabpfn(
    model: Any,
    y_test_original: np.ndarray,
    target_scale: TargetScale,
):
    """
    Oracle predictor that fits the TabPFN model directly on the ground truth targets of each individual test instance.

    Args:
        model (Any): The TabPFN model instance.
        y_test_original (np.ndarray): Unscaled ground truth targets of shape `(B, S)`, where `B` is the number of test instances and `S` is the number of samples per instance.
        target_scale (TargetScale): The scaling method to apply to the targets prior to fitting.

    Returns:
        Tuple[List[dict], Any]:
            - A list of length `B` containing full prediction dictionaries (moved to CPU for memory efficiency).
            - The fitted target scaler (if `target_scale == TargetScale.MAX`), otherwise `None`.
    """
    y_scaler = None
    y_test_scaled = y_test_original
    if target_scale == TargetScale.LOG:
        y_test_scaled = log1p_scaling(y_test_original)[0]
    elif target_scale == TargetScale.MAX:
        y_test_scaled, y_scaler = max_scaling(y_test_original)

    tabpfn_preds = []

    for inst in y_test_scaled:
        # Create dummy features (zeros) for the current instance
        X_temp = np.zeros((inst.shape[0], 1))

        # Fit the model specifically on the targets for this instance
        model.fit(X_temp, inst)

        with torch.inference_mode():
            # Predict using the first dummy feature entry as the representative for this instance
            preds = model.predict(X_temp[:1], output_type="full")
            tabpfn_preds.append(dict_to_cpu(preds))

    return tabpfn_preds, y_scaler
