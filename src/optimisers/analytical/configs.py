"""
Goal: This file contains the configurations for gradient optimisation algorithms. Most of these will be pydantic models.
Context: The pydantic models will ensure users know how to set up their own models.
"""

# -------------------------------------------------------------------------------------------------------------------- #
# Standard Library
from typing import Optional, TypeVar

# Third Party
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, validator

# Private
from src.utils.enums import OptimiserType, OptimiserCategory


# -------------------------------------------------------------------------------------------------------------------- #
class Parameter(BaseModel):
    """Container for parameter and its gradient with validation."""

    data: NDArray[np.float64] = Field(description="Parameter values")
    grad: Optional[NDArray[np.float64]] = Field(
        default=None, description="Parameter gradients"
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
        validate_assignment = True

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


# -------------------------------------------------------------------------------------------------------------------- #
GradientConfigT = TypeVar("GradientConfigT", bound="BaseGradientOptimiserConfig")


class BaseGradientOptimiserConfig(BaseModel):
    """Base for gradient-based optimizers."""

    category: OptimiserCategory = Field(
        default=OptimiserCategory.ANALYTICAL, const=True
    )
    optimiser_type: OptimiserType
    lr: float = Field(gt=0, default=0.01, description="Learning rate.")


class GradientDescentConfig(BaseGradientOptimiserConfig):
    """Gradient descent configuration."""

    optimiser_type: OptimiserType = Field(
        default=OptimiserType.GRADIENT,
        const=True,  # Can't be changed
        description="Type of optimiser involvement.",
    )
    lr: float = Field(gt=0, default=0.01, description="Learning rate.")


class AdamConfig(BaseGradientOptimiserConfig):
    """Adam optimizer configuration."""

    optimiser_type: OptimiserType = Field(
        default=OptimiserType.ADAM,
        const=True,
        description="Type of optimiser involvement.",
    )
    lr: float = Field(gt=0, default=0.01, description="Learning rate.")
    beta1: float = Field(
        default=0.9,
        ge=0,
        lt=1,
        description="Exponential decay rate for first moment estimate.",
    )
    beta2: float = Field(
        default=0.999,
        ge=0,
        lt=1,
        description="Exponential decay rate for second moment estimate.",
    )
    eps: float = Field(
        default=1e-8, gt=0, lt=1, description="Small constant for numerical stability."
    )
