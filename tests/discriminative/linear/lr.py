"""
Goal: This file runs various tests on the Linear Regression model.
Context: To test various dimenionsional datasets.
"""

# ------------------------------------------------------------------------------------ #
# Standard Library

# Third Party
import numpy as np

# Private
from src.discriminative.linear.lr import LinearRegression
from src.optimisers.gd import GD
from src.optimisers.trainer import Trainer
from src.utils.configs import GDConfig, TrainerConfig


# ------------------------------------------------------------------------------------ #
def test_lr_gradient_based():
    # Arrange
    np.random.seed(42)
    X = np.random.randn(100, 3)
    true_weights = np.array([1.0, 2.0, 3.0])
    y = X @ true_weights + 0.5 + np.random.randn(100) * 0.1
    # Act
    model = LinearRegression(input_dim=3)
    gd_config = GDConfig(lr=0.01)
    optimiser = GD(model.parameters(), config=gd_config)
    trainer = Trainer(TrainerConfig(batch_size=32, epochs=100, verbose=True))
    trainer.train(model=model, X=X, y=y, optimiser=optimiser, loss_fn=...)
    y_pred = model.predict(X)
    # Assert
    assert len(y_pred) == len(y)
