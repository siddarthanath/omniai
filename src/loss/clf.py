"""
Goal: This file contains the loss functions for classification problems.
Context: Classification problems have a discrete target within the loss function.
"""

# -------------------------------------------------------------------------------------------------------------------- #
# Standard Library
from typing import ClassVar

# Third Party
import numpy as np

# Private
from src.utils.enums import MLTaskType


# -------------------------------------------------------------------------------------------------------------------- #
class CrossEntropyLoss:
    ml_task: ClassVar[MLTaskType] = MLTaskType.CLASSIFICATION

    def __init__(self, eps: float = 1e-15):
        self.eps = eps

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # Clip predictions for numerical stability
        y_pred_safe = np.clip(y_pred, self.eps, 1 - self.eps)
        # Binary case
        if y_pred.ndim == 1:
            return -np.mean(
                y_true * np.log(y_pred_safe) + (1 - y_true) * np.log(1 - y_pred_safe)
            )
        # Multi-class case
        else:
            return -np.mean(np.sum(y_true * np.log(y_pred_safe), axis=1))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_pred_safe = np.clip(y_pred, self.eps, 1 - self.eps)
        # Binary case
        if y_pred.ndim == 1:
            return ((y_pred_safe - y_true) / (y_pred_safe * (1 - y_pred_safe))) / len(
                y_pred
            )
        # Multi-class case
        else:
            return -y_true / y_pred_safe / len(y_pred)
