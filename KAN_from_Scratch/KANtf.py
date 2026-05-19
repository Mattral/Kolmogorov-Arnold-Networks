import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


class KANLinear(Layer):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        activation="silu",
        regularization_factor=0.01,
        grid_range=(-1, 1),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.activation_func = getattr(tf.nn, activation)
        self.regularizer = l2(regularization_factor)
        self.grid_range = grid_range

        # Initialize weights
        # Number of B-spline basis functions: grid_size + spline_order when using extended grid
        num_basis_functions = grid_size + spline_order
        self.base_weight = self.add_weight(
            name="base_weight",
            shape=(in_features, out_features),
            initializer="glorot_uniform",
            regularizer=self.regularizer,
            trainable=True,
        )
        self.spline_weight = self.add_weight(
            name="spline_weight",
            shape=(in_features, out_features, num_basis_functions),
            initializer="glorot_uniform",
            regularizer=self.regularizer,
            trainable=True,
        )

        self.build_grid()

    def build_grid(self):
        # Initialize a one-dimensional grid of points
        initial_grid = np.linspace(
            self.grid_range[0], self.grid_range[1], self.grid_size
        ).astype(np.float32)
        self.grid = self.add_weight(
            name="grid",
            shape=(self.grid_size,),
            initializer=tf.constant_initializer(initial_grid),
            trainable=True,
        )

    def call(self, inputs):
        base_output = tf.matmul(inputs, self.base_weight)
        spline_output = self.compute_spline_output(inputs)
        return self.activation_func(base_output + spline_output)

    def compute_spline_output(self, inputs):
        """Compute spline contribution to the output.

        Args:
            inputs: Tensor of shape (batch_size, in_features)

        Returns:
            Tensor of shape (batch_size, out_features)
        """
        # Compute B-spline basis for each input feature
        # inputs shape: (batch_size, in_features)
        # We need to evaluate splines for each feature dimension
        batch_size = tf.shape(inputs)[0]

        # Initialize output
        spline_output = tf.zeros([batch_size, self.out_features], dtype=tf.float32)

        # For each input feature, compute its spline contribution
        for i in range(self.in_features):
            # Extract the i-th feature for all samples: (batch_size,)
            x_i = inputs[:, i:i + 1]  # (batch_size, 1)

            # Compute B-spline basis values for this feature
            # Result shape: (batch_size, num_basis_functions)
            b_spline_basis = B_batch_tf(
                x_i, self.grid, k=self.spline_order, extend=True
            )

            # Apply spline weights and sum
            # spline_weight shape: (in_features, out_features, num_basis_functions)
            # b_spline_basis shape: (batch_size, num_basis_functions)
            # We want output shape: (batch_size, out_features)
            num_basis = tf.shape(b_spline_basis)[1]
            weights_i = self.spline_weight[
                i, :, :num_basis
            ]  # (out_features, num_basis)

            # Compute weighted sum: (batch_size, num_basis) @ (num_basis, out_features)^T
            contribution = tf.matmul(
                b_spline_basis, tf.transpose(weights_i)
            )  # (batch_size, out_features)
            spline_output += contribution

        return spline_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_features": self.in_features,
                "out_features": self.out_features,
                "grid_size": self.grid_size,
                "spline_order": self.spline_order,
                "activation": self.activation_func.__name__,
                "regularization_factor": self.regularizer.l2.numpy,
                "grid_range": self.grid_range,
            }
        )
        return config


def extend_grid_tf(grid, k):
    if tf.rank(grid) != 1:
        raise ValueError("Grid tensor must be one-dimensional")
    left = tf.fill([k], 2 * grid[0] - grid[k])  # symmetric extension at the start
    right = tf.fill([k], 2 * grid[-1] - grid[-k - 1])  # symmetric extension at the end
    return tf.concat([left, grid, right], axis=0)


def B_batch_tf(x, grid, k=3, extend=True):
    """
    Compute B-spline basis values for given inputs using TensorFlow.

    Simplified recursive implementation of the de Boor-Cox formula.

    Args:
        x : Tensor
            Input values, shape (num_samples, 1).
        grid : Tensor
            Grid points, shape (num_grid_points,).
        k : int
            Order of the B-spline (degree is k-1).
        extend : bool
            If True, extends the grid by k points on both ends to handle boundary conditions.

    Returns:
        Tensor
            B-spline basis values.
    """
    # Ensure grid is 1D
    if len(grid.shape) != 1:
        raise ValueError(f"Grid must be 1D, got shape {grid.shape}")

    if extend:
        grid = extend_grid_tf(grid, k)

    # Clamp x to valid range
    x = tf.clip_by_value(x, grid[0], grid[-1] - 1e-10)

    # Compute B-spline basis using Cox-de Boor recursion
    # Start with piecewise constant basis functions
    grid_len = tf.shape(grid)[0]
    num_basis = grid_len - 1

    # Reshape grid for broadcasting: (num_basis,) for left endpoints
    grid_left = grid[:num_basis]  # grid[0:num_basis]
    grid_right = grid[1:grid_len]  # grid[1:grid_len]

    # Initialize: B_i^0(x) = 1 if grid[i] <= x < grid[i+1], else 0
    # Shape: (num_samples, num_basis)
    basis = tf.cast(
        tf.logical_and(
            x >= tf.reshape(grid_left, [1, -1]), x < tf.reshape(grid_right, [1, -1])
        ),
        tf.float32,
    )

    # Recursively compute higher order basis functions
    for p in range(1, k):
        # Each iteration reduces the number of basis functions by 1
        num_basis_new = grid_len - 1 - p

        # Create slices of grid for this iteration
        # For order p, we use grid[i:i+p+1] for each basis function i
        grid_left_p = grid[:num_basis_new]  # grid[0:num_basis_new]
        grid_left_p_plus_p = grid[p:p + num_basis_new]  # grid[p:p+num_basis_new]
        grid_right = grid[1:num_basis_new + 1]  # grid[1:num_basis_new+1]
        grid_right_p_plus_1 = grid[
            p + 1:p + num_basis_new + 1
        ]  # grid[p+1:p+num_basis_new+1]

        # Compute new basis functions
        # B_i^p(x) = (x - grid[i])/(grid[i+p] - grid[i]) * B_i^(p-1)(x)
        #          + (grid[i+p+1] - x)/(grid[i+p+1] - grid[i+1]) * B_{i+1}^(p-1)(x)

        denom_left = grid_left_p_plus_p - grid_left_p
        denom_right = grid_right_p_plus_1 - grid_right

        # Avoid division by zero
        denom_left = tf.where(denom_left == 0, tf.ones_like(denom_left), denom_left)
        denom_right = tf.where(denom_right == 0, tf.ones_like(denom_right), denom_right)

        # Compute weighted sums
        # basis shape: (num_samples, num_basis)
        # We need: (num_samples, num_basis_new)
        left_weight = (x - tf.reshape(grid_left_p, [1, -1])) / tf.reshape(
            denom_left, [1, -1]
        )
        right_weight = (tf.reshape(grid_right_p_plus_1, [1, -1]) - x) / tf.reshape(
            denom_right, [1, -1]
        )

        # basis[:, :num_basis_new] * left_weight + basis[:, 1:num_basis_new+1] * right_weight
        basis = (
            left_weight * basis[:, :num_basis_new]
            + right_weight * basis[:, 1:num_basis_new + 1]
        )

    return basis


class KAN(tf.keras.models.Sequential):
    def __init__(self, layers_configurations, **kwargs):
        super().__init__()
        for layer_config in layers_configurations:
            self.add(KANLinear(**layer_config, **kwargs))


def get_activations(model, model_inputs, layer_name=None):
    layer_outputs = [
        layer.output
        for layer in model.layers
        if layer.name == layer_name or layer_name is None
    ]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(model_inputs)
    return activations
