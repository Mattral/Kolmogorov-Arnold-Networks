# STILL NEED TO RESOLVE ISSUE and I am working on it ( feel free to fix )

import unittest
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

from KANtf import KANLinear, KAN, extend_grid_tf, B_batch_tf

class TestKANLinear(unittest.TestCase):
    def test_initialization(self):
        """Test if KANLinear layer initializes with correct shapes and configurable parameters."""
        in_features = 10
        out_features = 5
        grid_size = 5
        spline_order = 3
        layer = KANLinear(in_features, out_features, grid_size, spline_order)
        
        self.assertEqual(layer.in_features, in_features)
        self.assertEqual(layer.out_features, out_features)
        self.assertEqual(layer.grid_size, grid_size)
        self.assertEqual(layer.spline_order, spline_order)
        self.assertEqual(layer.base_weight.shape, (in_features, out_features))
        self.assertEqual(layer.spline_weight.shape, (in_features, out_features, grid_size + spline_order - 1))

    def test_forward_pass(self):
        """Test the forward pass computation of KANLinear."""
        layer = KANLinear(10, 5)
        input_tensor = tf.random.normal([10, 10])  # batch size of 10, 10 features
        output = layer(input_tensor)
        self.assertEqual(output.shape, (10, 5))

class TestBSplineFunctions(unittest.TestCase):
    def test_extend_grid(self):
        """Test if the grid is extended correctly on both sides."""
        grid = tf.constant([0.0, 1.0, 2.0])  # ensure it's 1D as expected
        extended_grid = extend_grid_tf(grid, 1)
        expected_output = [-1.0, 0.0, 1.0, 2.0, 3.0]
        np.testing.assert_array_almost_equal(extended_grid.numpy(), expected_output)

    def test_b_spline_basis(self):
        """Test B-spline basis computation for known inputs and grid."""
        x = tf.constant([[0.5], [1.5], [2.5]])
        grid = tf.constant([0.0, 1.0, 2.0, 3.0])  # changed shape
        b_spline_values = B_batch_tf(x, tf.expand_dims(grid, 0), k=2, extend=False)  # ensure grid dimensions are expanded
        expected_shape = (1, 3, 3)  # (num_splines, num_samples, num_grid_points + k - 1)
        self.assertEqual(b_spline_values.shape, expected_shape)


class TestKANModel(unittest.TestCase):
    def test_model_construction(self):
        """Test the construction of the KAN model."""
        layers_config = [
            {'in_features': 10, 'out_features': 5},
            {'in_features': 5, 'out_features': 3}
        ]
        model = KAN(layers_configurations=layers_config)
        self.assertIsInstance(model, tf.keras.models.Sequential)
        self.assertEqual(len(model.layers), 2)

if __name__ == '__main__':
    unittest.main()
