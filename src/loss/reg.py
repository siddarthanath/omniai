"""
Goal: This file contains the loss functions for regression problems.
Context: Classification problems have a continuous target within the loss function.
"""

# ------------------------------------------------------------------------------------ #
# Standard Library
from typing import ClassVar

# Third Party
import numpy as np
# Private
from src.utils.enums import MLTaskType


# ------------------------------------------------------------------------------------ #
class MSELoss:
    ml_task: ClassVar[MLTaskType] = MLTaskType.REGRESSION

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Mean squared error objective."""
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def backward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Derivative of MSE."""
        return - (2 / len(y_pred)) * (y_true - y_pred)
