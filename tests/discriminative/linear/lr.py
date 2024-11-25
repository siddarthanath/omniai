"""
Goal: This file runs various tests on the Linear Regression model.
Context: To test various dimenionsional datasets.
"""

# ------------------------------------------------------------------------------------ #
# Standard Library

# Third Party
import numpy as np
import matplotlib.pyplot as plt
import pytest
import plotly.graph_objects as go

# Private
from src.discriminative.linear.lr import LinearRegression
from src.optimisers.gd import GD
from src.optimisers.trainer import Trainer
from src.utils.configs import GDConfig, TrainerConfig
from src.loss.reg import MSELoss


# ------------------------------------------------------------------------------------ #
@pytest.mark.parametrize(
    "test_case,X_fn,add_noise",
    [
        ("gaussian_no_noise", lambda n: np.linspace(-3, 3, n).reshape(-1, 1), False),
        ("gaussian_with_noise", lambda n: np.linspace(-3, 3, n).reshape(-1, 1), True),
        ("random_no_noise", lambda n: np.random.randn(n, 1), False),
        ("random_with_noise", lambda n: np.random.randn(n, 1), True),
    ]
)
def test_lr_2d(test_case, X_fn, add_noise):

    # Arrange
    n_samples = 100
    X = X_fn(n_samples)
    X_standardized = (X - X.mean()) / X.std()
    true_weights = np.array([2.785969])
    noise = np.random.randn(n_samples) * 0.1 if add_noise else 0
    y = X_standardized @ true_weights + 0.5 + noise

    # Act
    model = LinearRegression(input_dim=1)
    optimiser = GD(model.parameters(), config=GDConfig(lr=0.1))
    trainer = Trainer(TrainerConfig(batch_size=32, epochs=100))
    trainer.train(model=model, X=X_standardized, y=y, optimiser=optimiser, loss_fn=MSELoss())

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X_standardized.flatten(),
        y=y,
        mode='markers',
        name='Data Points'
    ))
    X_sorted = np.sort(X_standardized, axis=0)
    y_pred = model.predict(X_sorted)
    fig.add_trace(go.Scatter(
        x=X_sorted.flatten(),
        y=y_pred,
        mode='lines',
        name='Fitted Line'
    ))
    fig.update_layout(
        title=f'Linear Regression: {test_case}',
        xaxis_title='X (standardized)',
        yaxis_title='y'
    )
    fig.show()
    print(f'{test_case}:\nTrue weights: {true_weights}\nEstimated: w={model.weights.data}, b={model.bias.data}')


@pytest.mark.parametrize(
    "test_case,X_fn,add_noise",
    [
        ("gaussian_no_noise", lambda n: np.column_stack([np.linspace(-3, 3, n), np.linspace(-2, 2, n)]), False),
        ("gaussian_with_noise", lambda n: np.column_stack([np.linspace(-3, 3, n), np.linspace(-2, 2, n)]), True),
        ("random_no_noise", lambda n: np.random.randn(n, 2), False),
        ("random_with_noise", lambda n: np.random.randn(n, 2), True),
    ]
)
def test_lr_3d(test_case, X_fn, add_noise):
    # Arrange
    n_samples = 100
    X = X_fn(n_samples)
    X_standardized = (X - X.mean(axis=0)) / X.std(axis=0)
    true_weights = np.array([2.5, 1.5])
    noise = np.random.randn(n_samples) * 0.1 if add_noise else 0
    y = X_standardized @ true_weights + 0.5 + noise

    # Act
    model = LinearRegression(input_dim=2)
    optimiser = GD(model.parameters(), config=GDConfig(lr=0.1))
    trainer = Trainer(TrainerConfig(batch_size=32, epochs=100))
    trainer.train(model=model, X=X_standardized, y=y, optimiser=optimiser, loss_fn=MSELoss())

    # Plot
    x1_mesh, x2_mesh = np.meshgrid(
        np.linspace(X_standardized[:, 0].min(), X_standardized[:, 0].max(), 20),
        np.linspace(X_standardized[:, 1].min(), X_standardized[:, 1].max(), 20)
    )
    X_mesh = np.column_stack([x1_mesh.ravel(), x2_mesh.ravel()])
    y_mesh = model.predict(X_mesh).reshape(x1_mesh.shape)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=X_standardized[:, 0],
        y=X_standardized[:, 1],
        z=y,
        mode='markers',
        marker=dict(size=5, opacity=0.6)
    ))
    fig.add_trace(go.Surface(
        x=x1_mesh,
        y=x2_mesh,
        z=y_mesh,
        opacity=0.5
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title='X1',
            yaxis_title='X2',
            zaxis_title='y'
        ),
        title=f'Linear Regression 3D: {test_case}'
    )
    fig.show()
    print(f'{test_case}:\nTrue weights: {true_weights}\nEstimated: w={model.weights.data}, b={model.bias.data}')

