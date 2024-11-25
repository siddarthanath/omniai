"""
Goal: This file contains the enums for the library.
Context: Enums make it useful to understand categorical variables.
"""

# ------------------------------------------------------------------------------------ #
# Standard Library
from enum import Enum


# Third Party

# Private


# ------------------------------------------------------------------------------------ #
class MLTaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
