"""
Goal: This file contains the Linear Regression model.
Context: Linear Regression has the following form of f(x) = w_0+w_1x_1+...+w_nx_n, which
can be writte as f(x) = Y = Xw + w_0 (or you can merge the bias into the w). Using
calculus, this model has the objective function of mean squared error (MSE), which
when minimised, gives us the optimal model parameters (via the normal equations).
Similarly, when using statistics and modelling the feature set as a Gaussian
distribution, the objective function is the negative log likelihood (NLL), which when
minimised, is equivalent to minimising the MSE.
"""

# ------------------------------------------------------------------------------------ #
# Standard Library
from typing import Optional

# Third Party
import numpy as np

# Private
from src.discriminative.base import DiscriminativeModel
from src.utils.configs import Parameter


# ------------------------------------------------------------------------------------ #
class LinearRegression(DiscriminativeModel):
    """Linear Regression model with both analytical and gradient-based solutions."""

    def __init__(self, input_dim: int, fit_intercept: bool = True):
        """Initialize LinearRegression model.

        Args:
            input_dim: Number of input features
            fit_intercept: Whether to include bias term
        """
        super().__init__()  # This initializes self._is_fitted to False
        self.input_dim = input_dim
        self.fit_intercept = fit_intercept

        # Initialize parameters using Pydantic model
        self.weights = Parameter(data=np.zeros(input_dim))
        self.bias = Parameter(data=np.zeros(1)) if fit_intercept else None

        # Training history
        self.loss_history: list[float] = []

    def parameters(self) -> list[Parameter]:
        """Get all trainable parameters."""
        if self.bias is not None:
            return [self.weights, self.bias]
        return [self.weights]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass of the model.

        Args:
            X: Input features of shape (n_samples, input_dim)

        Returns:
            Model predictions of shape (n_samples,)

        Raises:
            ValueError: If model is not fitted or weights are not initialized
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        if self.weights.data is None:
            raise ValueError("Model parameters not initialized")

        output = X @ self.weights.data
        if self.bias is not None:
            output += self.bias.data
        return output

    def backward(self, X: np.ndarray, grad_output: np.ndarray) -> None:
        """Backward pass to compute gradients.

        Args:
            X: Input features of shape (n_samples, input_dim)
            grad_output: Gradient of loss with respect to output

        Note:
            Updates gradients in-place for weights and bias parameters
        """
        # Average over batch dimension for proper scaling
        batch_size = len(X)

        # Compute gradients for weights (∂L/∂w = X.T @ grad_output / batch_size)
        self.weights.grad = X.T @ grad_output / batch_size

        # Compute gradients for bias (∂L/∂b = mean(grad_output))
        if self.bias is not None:
            self.bias.grad = np.mean(grad_output, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Fit model using analytical solution (normal equation).

        Args:
            X: Training features of shape (n_samples, input_dim)
            y: Target values of shape (n_samples,)

        Returns:
            self: Fitted model

        Note:
            Uses the normal equation: β = (X.T @ X)^(-1) @ X.T @ y
        """
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input features with {self.input_dim} dimensions, "
                f"got {X.shape[1]} dimensions"
            )

        # Add intercept column if needed
        X_design = self._add_intercept(X)

        # Solve normal equation
        try:
            beta = np.linalg.solve(X_design.T @ X_design, X_design.T @ y)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if matrix is singular
            beta = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y

        # Extract weights and bias
        if self.fit_intercept:
            self.bias.data = beta[0:1]  # type: ignore
            self.weights.data = beta[1:]
        else:
            self.weights.data = beta

        # Mark model as fitted
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for given features.

        Args:
            X: Input features of shape (n_samples, input_dim)

        Returns:
            Predictions of shape (n_samples,)

        Raises:
            ValueError: If model is not fitted
        """
        return self.forward(X)

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add intercept column to input features if needed.

        Args:
            X: Input features of shape (n_samples, input_dim)

        Returns:
            Array with intercept column if fit_intercept is True
        """
        if self.fit_intercept:
            return np.c_[np.ones(len(X)), X]
        return X

    @property
    def coef_(self) -> np.ndarray:
        """Get model coefficients.

        Raises:
            ValueError: If model is not fitted
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before accessing coefficients")
        return self.weights.data

    @property
    def intercept_(self) -> Optional[float]:
        """Get model intercept.

        Raises:
            ValueError: If model is not fitted
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before accessing intercept")
        if self.bias is not None:
            return float(self.bias.data[0])
        return None
