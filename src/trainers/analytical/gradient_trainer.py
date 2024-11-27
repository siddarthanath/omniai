"""
Goal: This file contains the Trainer algorithm used across all models.
Context: The main focus of this Trainer is for Gradient Based optimisation. This allows for any model to be trained,
even if an analytical solution does not exist.
"""

# -------------------------------------------------------------------------------------------------------------------- #
# Standard Library

# Third Party
import numpy as np

# Private
from src.optimisers.analytical.base import GradientOptimiser
from src.trainers.analytical.configs import GradientTrainerConfig
from src.utils.protocols import GradientModel, LossFunction
from src.trainers.base import BaseTrainer


# -------------------------------------------------------------------------------------------------------------------- #


class GradientTrainer(BaseTrainer[GradientOptimiser]):
    """Generic trainer for gradient-based optimization."""

    def __init__(self, config: GradientTrainerConfig, optimiser: GradientOptimiser):
        super().__init__(config, optimiser)
        self.config = config
        self.loss_history: list[float] = []
        self.optimiser = optimiser

    def train(
        self,
        model: GradientModel,
        X: np.ndarray,
        y: np.ndarray,
        loss_fn: LossFunction,
    ):
        """Train a model using Gradient-Based optimisation.

        Args:
            model (Model): Any machine learning model.
            X (np.ndarray): Feature set.
            y (np.ndarray): Target set.
            loss_fn (LossFunction): The loss function.

        Returns:
            list[float]: _description_
        """
        # Store number of samples
        n_samples = len(X)
        # Start epoch loop
        for epoch in range(self.config.epochs):
            # Stochastic
            indices = (
                np.random.permutation(n_samples)
                if self.config.shuffle
                else np.arange(n_samples)
            )
            # Mini-Batch
            total_loss = 0.0
            for start_idx in range(0, n_samples, self.config.batch_size):
                # Obtain batches
                batch_idx = indices[start_idx : start_idx + self.config.batch_size]
                # Train over batches and retrieve loss
                loss = self._train_batch(
                    model=model,
                    X=X[batch_idx],
                    y=y[batch_idx],
                    loss_fn=loss_fn,
                )
                # Obtain cumulative loss
                total_loss += loss * len(batch_idx)
            # Calculate average loss
            avg_loss = total_loss / n_samples
            # Store loss
            self.loss_history.append(avg_loss)
            # Log results
            if self.config.verbose and (epoch + 1) % self.config.verbose_freq == 0:
                print(f"Epoch {epoch + 1}/{self.config.epochs}, Loss: {avg_loss:.6f}")

            # @TODO: Add convergence criterion i.e., early stopping.

    def _train_batch(
        self,
        model: GradientModel,
        X: np.ndarray,
        y: np.ndarray,
        loss_fn: LossFunction,
    ) -> float:
        """Apply training step.

        Args:
            model (Model): Any machine learning model.
            X (np.ndarray): Feature set.
            y (np.ndarray): Target set.
            loss_fn (LossFunction): The loss function.

        Returns:
            float: The loss for the specific batch.
        """
        # Clear gradients
        self.optimiser.zero_grad()
        # Activate a full model pass (from input to output)
        y_pred = model.forward(X)
        # Calculate loss
        loss = loss_fn(y_pred, y)
        # Backpropagation i.e., calculate parameter gradients w.r.t loss
        grad_output = loss_fn.backward(y_pred, y)
        model.backward(X, grad_output)
        # Update model parameters
        self.optimiser.step()
        return loss

    def log_progress(self) -> None:
        """Log training progress."""
        pass

    def is_converged(self) -> bool:
        """Check if training has converged."""
        pass
