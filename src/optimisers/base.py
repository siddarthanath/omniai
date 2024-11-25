"""
Goal: This file contains the base class for optimisers.
Context: Optimisers are used to move the model parameters in the direction of the 
gradient of the loss function, which leads to objective minimisation.
"""

# ------------------------------------------------------------------------------------ #
# Standard Library
# Third Party
from abc import ABC, abstractmethod
from typing import Generic

# Private
from src.utils.configs import ConfigT, Parameter

# ------------------------------------------------------------------------------------ #


class Optimiser(ABC, Generic[ConfigT]):
    """Base optimizer class parameterized by config type."""

    def __init__(self, params: list[Parameter], config: ConfigT):
        self.params = params
        self.config = config

    def zero_grad(self) -> None:
        """Set all parameter gradients to zero."""
        for param in self.params:
            if param.grad is not None:
                param.grad.fill(0)

    @abstractmethod
    def step(self) -> None:
        """Update all parameters based on their gradients.
        This method must be implemented by all optimizers to define
        how parameters are updated using their gradients.
        """
        raise NotImplementedError("All subclasses must inherit this method!")
