"""
Goal: This file contains the Adagrad optimiser.
Context: The Adagrad optimiser was introduced to overcome issues with standard Gradient Descent.
"""

# -------------------------------------------------------------------------------------------------------------------- #
# Standard Library
from typing import Generic

# Third Party
import numpy as np

# Private
from src.optimisers.analytical.configs import GradientConfigT, Parameter

# -------------------------------------------------------------------------------------------------------------------- #
