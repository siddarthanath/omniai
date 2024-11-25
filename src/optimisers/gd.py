"""
Goal: This file contains the Gradient Descent algorithm.
Context: This allows us to minimise or maximise a function.
"""

# ------------------------------------------------------------------------------------ #
# Standard Library

# Third Party
# Private
from src.optimisers.base import Optimiser
from src.utils.configs import GDConfig, Parameter

# ------------------------------------------------------------------------------------ #


class GD(Optimiser[GDConfig]):
    """Gradient Descent optimizer."""

    def __init__(self, params: list[Parameter], config: GDConfig):
        super().__init__(params, config)

    def step(self) -> None:
        """Perform one step of gradient descent."""
        for param in self.params:
            if param.grad is not None:
                param.data -= self.config.lr * param.grad
