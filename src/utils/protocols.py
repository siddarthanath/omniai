"""
Goal: This file contains the protocols which defines the structure of classes.
Context: Similar to typing, protocols allow us to define what something 'looks' like,  without needing to know its
origin e.g., we have multiple discriminative models, each which follow the same structure so if we were to pass a
generic 'Model' parameter, the function will know its signature (regardless of the type of model it is).
"""

# -------------------------------------------------------------------------------------------------------------------- #
# Standard Library
from typing import Protocol, ClassVar

# Third Party
import numpy as np

# Private
from src.utils.enums import MLTaskType

# -------------------------------------------------------------------------------------------------------------------- #


class GradientModel(Protocol):
    """Protocol for models (with gradients) that can be trained."""

    def forward(self, X: np.ndarray) -> np.ndarray: ...

    def backward(self, X: np.ndarray, grad_output: np.ndarray) -> None: ...

    def parameters(self) -> list: ...


class LossFunction(Protocol):
    """Protocol for loss functions."""

    ml_task: ClassVar[MLTaskType]

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float: ...

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray: ...
