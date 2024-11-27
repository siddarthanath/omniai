"""
Goal: This file contains the base class for trainers.
Context: Trainers allow for any model to be trained, even if an analytical solution
does not exist.
"""

# -------------------------------------------------------------------------------------------------------------------- #
# Standard Library
from abc import ABC, abstractmethod
from typing import Protocol, TypeVar, Generic

# Third Party
import numpy as np

# Private
from src.trainers.analytical.configs import BaseTrainerConfig
from src.utils.protocols import LossFunction

# -------------------------------------------------------------------------------------------------------------------- #
OptimiserT = TypeVar("OptimiserT")


class BaseTrainer(ABC, Generic[OptimiserT]):
    """Base class for all trainers."""

    def __init__(self, config: BaseTrainerConfig, optimiser: OptimiserT):
        self.config = config
        # Training history (loss, metrics etc.)
        self.history: dict = {}
        # Store optimiser
        self.optimiser = optimiser

    @abstractmethod
    def train(
        self,
        model: Protocol,
        X: np.ndarray,
        y: np.ndarray,
        loss_fn: LossFunction,
    ) -> dict:
        """Train a model.

        Args:
            model: Model to train.
            X: Feature set.
            y: Target set.

        Returns:
            Training history.
        """
        pass

    @abstractmethod
    def log_progress(self) -> None:
        """Log training progress."""
        raise NotImplementedError("All subclasses must implement this method!")

    @property
    @abstractmethod
    def is_converged(self) -> bool:
        """Check if training has converged."""
        raise NotImplementedError("All subclasses must implement this method!")
