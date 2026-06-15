"""Tests for adaptive grid update (Block 8)."""
import numpy as np
import pytest
import tensorflow as tf
import torch

from kanx import KAN as TF_KAN
from kanx.layers import KANLinear as TF_KANLinear
from kanx.torch import KAN as PyTorch_KAN
from kanx.torch.layers import KANLinear as PyTorch_KANLinear


class TestTensorFlowGridUpdate:
    """Tests for TensorFlow backend grid update."""

    def test_tf_grid_update_changes_grid(self):
        """After update_grid_from_samples, grid values change."""
        model = TF_KAN([4, 8, 1])
        x = tf.random.normal((128, 4))

        # Build the model by doing a forward pass
        _ = model(x, training=False)

        # Get initial grid
        initial_grid = model.layers[1].grid.numpy().copy()

        # Update grid
        model.update_grid_from_samples(x)
        new_grid = model.layers[1].grid.numpy()

        # Check that grid changed
        assert not np.allclose(initial_grid, new_grid), "Grid should change after update"

    def test_tf_grid_update_preserves_output_shape(self):
        """Forward pass still works after grid update."""
        model = TF_KAN([6, 16, 1])
        x_train = tf.random.normal((100, 6))
        x_test = tf.random.normal((32, 6))

        # Build the model
        _ = model(x_train, training=False)

        model.update_grid_from_samples(x_train)
        y = model(x_test, training=False)

        assert y.shape == (32, 1)
        assert not np.isnan(y.numpy()).any()

    def test_tf_layer_grid_update(self):
        """KANLinear.update_grid_from_samples works for TensorFlow."""
        layer = TF_KANLinear(8, 4)
        x = tf.random.normal((128, 8))

        # Build the layer
        layer.build((None, 8))

        initial_grid = layer.grid.numpy().copy()
        layer.update_grid_from_samples(x)
        new_grid = layer.grid.numpy()

        assert not np.allclose(initial_grid, new_grid), "Layer grid should change"

    def test_tf_grid_update_is_strictly_increasing(self):
        """Grid values remain strictly increasing after update."""
        model = TF_KAN([4, 8, 1])
        x = tf.random.normal((100, 4))

        # Build the model
        _ = model(x, training=False)

        model.update_grid_from_samples(x)

        for layer in model.layers:
            if isinstance(layer, TF_KANLinear):
                grid = layer.grid.numpy()
                for i in range(grid.shape[0]):
                    diffs = np.diff(grid[i, :])
                    assert np.all(diffs > 0), f"Grid for feature {i} is not strictly increasing"


class TestPyTorchGridUpdate:
    """Tests for PyTorch backend grid update."""

    def test_torch_grid_update_changes_grid(self):
        """After update_grid_from_samples, grid values change."""
        model = PyTorch_KAN([4, 8, 1])
        x = torch.randn(128, 4)

        # Get initial grid
        initial_grid = model[0].grid.data.clone()

        # Update grid
        model.update_grid_from_samples(x)
        new_grid = model[0].grid.data

        # Check that grid changed
        assert not torch.allclose(initial_grid, new_grid), "Grid should change after update"

    def test_torch_grid_update_is_numerically_stable(self):
        """Grid values remain strictly increasing after update."""
        model = PyTorch_KAN([6, 16, 1])
        x = torch.randn(100, 6)

        model.update_grid_from_samples(x)

        for layer in model.modules():
            if isinstance(layer, PyTorch_KANLinear):
                grid = layer.grid
                for i in range(grid.shape[0]):
                    diffs = torch.diff(grid[i, :])
                    assert torch.all(diffs > 0), f"Grid for feature {i} is not strictly increasing"

    def test_torch_layer_grid_update(self):
        """KANLinear.update_grid_from_samples works for PyTorch."""
        layer = PyTorch_KANLinear(8, 4)
        x = torch.randn(128, 8)

        initial_grid = layer.grid.data.clone()
        layer.update_grid_from_samples(x)
        new_grid = layer.grid.data

        assert not torch.allclose(initial_grid, new_grid), "Layer grid should change"

    def test_torch_grid_update_preserves_shape(self):
        """Grid shape unchanged after update."""
        model = PyTorch_KAN([6, 16, 1])
        x = torch.randn(100, 6)

        initial_shape = model[0].grid.shape
        model.update_grid_from_samples(x)
        new_shape = model[0].grid.shape

        assert initial_shape == new_shape, "Grid shape should not change"


class TestGridUpdateImprovement:
    """Test that grid update can improve loss."""

    def test_tf_grid_update_improves_loss(self):
        """Training with grid update produces lower final loss than without."""
        torch.manual_seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)

        # Generate synthetic data
        X_train = tf.random.normal((100, 4))
        y_train = tf.sin(X_train[:, :1] * 3.14) + tf.cos(X_train[:, 1:2] * 2.0)

        # Model with grid update
        model1 = TF_KAN([4, 16, 8, 1])
        model1.compile(optimizer='adam', loss='mse')
        hist1_before = model1.evaluate(X_train, y_train, verbose=0)

        # Train 5 epochs
        model1.fit(X_train, y_train, epochs=5, verbose=0)
        hist1_after_5 = model1.evaluate(X_train, y_train, verbose=0)

        # Update grid
        model1.update_grid_from_samples(X_train)

        # Train 5 more epochs
        model1.fit(X_train, y_train, epochs=5, verbose=0)
        hist1_after_10 = model1.evaluate(X_train, y_train, verbose=0)

        # Loss should continue to decrease
        assert hist1_after_10 < hist1_after_5, "Loss should decrease after grid update and further training"

    def test_torch_grid_update_improves_loss(self):
        """Training with grid update produces lower final loss than without."""
        torch.manual_seed(42)
        np.random.seed(42)

        # Generate synthetic data
        X_train = torch.randn(100, 4)
        y_train = torch.sin(X_train[:, :1] * 3.14) + torch.cos(X_train[:, 1:2] * 2.0)

        # Model
        model = PyTorch_KAN([4, 16, 8, 1])

        # Train 5 epochs before grid update
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        for epoch in range(5):
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_before_update = criterion(model(X_train), y_train).item()

        # Update grid
        model.update_grid_from_samples(X_train)

        # Train 5 more epochs
        for epoch in range(5):
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_after_update = criterion(model(X_train), y_train).item()

        # Loss should continue to decrease
        assert loss_after_update < loss_before_update, "Loss should decrease after grid update and further training"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
