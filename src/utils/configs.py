"""
Goal: This file contains the configurations for optimisation algorithms. Most of these
will be pydantic models.
Context: The pydantic models will ensure users know how to setup their own models.
"""

# ------------------------------------------------------------------------------------ #
# Standard Library
from typing import Literal, Optional, Protocol, TypeVar

# Third Party
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, validator

# Private


# ------------------------------------------------------------------------------------ #
class Parameter(BaseModel):
    """Container for parameter and its gradient with validation."""

    data: NDArray[np.float64] = Field(description="Parameter values")
    grad: Optional[NDArray[np.float64]] = Field(
        default=None, description="Parameter gradients"
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
        validate_assignment = True  # Validate when attributes are set

    @validator("grad")
    def validate_grad_shape(cls, v, values):
        """Validate that gradient shape matches parameter shape."""
        if v is not None and "data" in values:
            if v.shape != values["data"].shape:
                raise ValueError(
                    f"Gradient shape {v.shape} must match "
                    f"parameter shape {values['data'].shape}"
                )
        return v


# ------------------------------------------------------------------------------------ #


class TrainerConfig(BaseModel):
    """Configuration for training."""

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


# ------------------------------------------------------------------------------------ #
ConfigT = TypeVar("ConfigT", bound="BaseOptimiserConfig")


class BaseOptimiserConfig(BaseModel):
    """Generic attribute in all optimiser configs."""

    lr: float


# ------------------------------------------------------------------------------------ #


class GDConfig(BaseOptimiserConfig):
    """Gradient Descent configuration."""

    lr: float = Field(gt=0, default=0.01, description="Learning rate")
    optimiser_type: Literal["gd"] = "gd"
