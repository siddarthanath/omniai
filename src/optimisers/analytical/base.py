"""
Goal: This file contains the base class for Gradient optimisers.
Context: Optimisers are used to move the model parameters in the direction of the  gradient of the loss function, which
leads to objective minimisation.
"""

# -------------------------------------------------------------------------------------------------------------------- #
# Standard Library
from typing import Generic

# Third Party
import numpy as np

# Private
from src.optimisers.analytical.configs import GradientConfigT, Parameter

# -------------------------------------------------------------------------------------------------------------------- #


class GradientOptimiser(Generic[GradientConfigT]):
    """Base gradient optimiser class parameterised by config type for gradient based optimisation."""

    def __init__(self, params: list[Parameter], config: GradientConfigT):
        self.params = params
        self.config = config

    @staticmethod
    def _compute_gradient_update(param: Parameter) -> np.ndarray:
        """Compute the gradient update. This is defaulted to generic gradient descent.
        Can be overridden by subclasses."""
        return param.grad

    def zero_grad(self) -> None:
        """Set all parameter gradients to zero."""
        for param in self.params:
            if param.grad is not None:
                param.grad.fill(0)

    def step(self) -> None:
        """Update all parameters based on their gradients.
        This method must be implemented by all optimizers to define
        how parameters are updated using their gradients.
        """
        for param in self.params:
            if param.grad is not None:
                param.data -= self.config.lr * self._compute_gradient_update(param)
