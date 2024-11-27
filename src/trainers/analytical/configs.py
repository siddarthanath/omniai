"""
Goal: This file contains the configurations for trainer algorithms. Most of these will be pydantic models.
Context: The pydantic models will ensure users know how to set up their own models.
"""

# -------------------------------------------------------------------------------------------------------------------- #
# Standard Library
from typing import TypeVar

# Third Party
from pydantic import BaseModel, Field


# -------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------- #
TrainerConfigT = TypeVar("TrainerConfigT")


class BaseTrainerConfig(BaseModel):
    """Base configuration for all trainers."""

    pass


class GradientTrainerConfig(BaseTrainerConfig):
    """Configuration for training for gradient based optimisation."""

    batch_size: int = Field(gt=0, default=32, description="Batch size to train.")
    epochs: int = Field(
        gt=0, default=100, description="Number of full training iteration cycles."
    )
    shuffle: bool = Field(
        default=True,
        description="Whether to shuffle the training data (for stochastic optimisation).",
    )
    verbose: bool = Field(default=True, description="Whether to log results.")
    verbose_freq: int = Field(gt=0, default=10, description="...")
