"""
Goal: This file contains the enums for the library.
Context: Enums make it useful to understand categorical variables.
"""

# -------------------------------------------------------------------------------------------------------------------- #
# Standard Library
from enum import Enum


# Third Party

# Private


# -------------------------------------------------------------------------------------------------------------------- #
class MLTaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class OptimiserCategory(Enum):
    ANALYTICAL = "analytical"
    HEURISTIC = "heuristic"


class OptimiserType(Enum):
    # Analytical/Gradient-based
    GRADIENT = "gradient"
    ADAM = "adam"
    RMSPROP = "rmsprop"
    # Heuristic
    GENETIC = "genetic"
    BAYESIAN = "bayesian"
    ANNEALING = "annealing"
