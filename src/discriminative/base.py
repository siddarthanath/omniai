"""
Goal: This file contains the base class for Discriminative models.
Context: Discriminative models learn P(y|x) directly.
"""

# ------------------------------------------------------------------------------------ #
# Standard Library

# Third Party
from abc import ABC, abstractmethod

import numpy as np

# Private


# ------------------------------------------------------------------------------------ #


class DiscriminativeModel(ABC):
    """Base class for all discriminative models."""

    def __init__(self) -> None:
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DiscriminativeModel":
        """Fit the model to training data.

        Args:
            X: Feature matrix
            y: Target values

        Returns:
            self: The fitted model
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for given features.

        Args:
            X: Feature matrix

        Returns:
            predictions: Model predictions
        """
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate model score on given data.

        Args:
            X: Feature matrix
            y: True target values

        Returns:
            score: Model score (higher is better)
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }

    def set_params(self, **params) -> "DiscriminativeModel":
        """Set model parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        params = [f"{k}={v}" for k, v in self.get_params().items()]
        return f"{self.__class__.__name__}({', '.join(params)})"
