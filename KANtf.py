import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model

class KANLinear(Layer):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 activation='silu', regularization_factor=0.01, grid_range=(-1, 1), **kwargs):
        super(KANLinear, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.activation_func = getattr(tf.nn, activation)
        self.regularizer = l2(regularization_factor)
        self.grid_range = grid_range

        # Initialize weights
        self.base_weight = self.add_weight(
            "base_weight",
            shape=(in_features, out_features),
            initializer='glorot_uniform',
            regularizer=self.regularizer,
            trainable=True)
        self.spline_weight = self.add_weight(
            "spline_weight",
            shape=(in_features, out_features, grid_size + spline_order - 1),
            initializer='glorot_uniform',
            regularizer=self.regularizer,
            trainable=True)

        self.build_grid()

    def build_grid(self):
        grid_start, grid_end = self.grid_range
        grid_span = (grid_end - grid_start) / (self.grid_size - 1)
        grid_points = tf.linspace(grid_start, grid_end, self.grid_size)
        self.grid = self.add_weight(
            "grid",
            shape=(self.grid_size,),
            initializer=tf.constant_initializer(grid_points),
            trainable=False)

    def call(self, inputs):
        base_output = tf.matmul(inputs, self.base_weight)
        spline_output = self.compute_spline_output(inputs)
        return self.activation_func(base_output + spline_output)

    def compute_spline_output(self, inputs):
        # Placeholder for B-spline calculation logic
        inputs_expanded = tf.expand_dims(inputs, -1)
        
        # Assume a function B_batch is defined similarly as in PyTorch to compute B-spline basis functions
        b_spline_bases = B_batch_tf(inputs_expanded, self.grid, k=self.spline_order)

        spline_output = tf.einsum('bik,ijk->bij', b_spline_bases, self.spline_weight)
        return spline_output

    def get_config(self):
        config = super().get_config()
        config.update({
            'in_features': self.in_features,
            'out_features': self.out_features,
            'grid_size': self.grid_size,
            'spline_order': self.spline_order,
            'activation': self.activation_func.__name__,
            'regularization_factor': self.regularizer.l2.numpy,
            'grid_range': self.grid_range
        })
        return config

def extend_grid_tf(grid, k_extend):
    h = (grid[:, -1:] - grid[:, :1]) / (tf.shape(grid)[1] - 1)
    left_extended = grid[:, :1] - h * tf.range(1, k_extend + 1, dtype=grid.dtype)[:, tf.newaxis]
    right_extended = grid[:, -1:] + h * tf.range(1, k_extend + 1, dtype=grid.dtype)[:, tf.newaxis]
    
    extended_grid = tf.concat([left_extended, grid, right_extended], axis=1)
    return extended_grid

def B_batch_tf(x, grid, k=3, extend=True):
    """
    Compute B-spline basis values for given inputs using TensorFlow.
    
    Args:
    -----
        x : Tensor
            Input values, shape (num_samples, 1).
        grid : Tensor
            Grid points, shape (num_splines, num_grid_points).
        k : int
            Order of the B-spline (degree is k-1).
        extend : bool
            If True, extends the grid by k points on both ends to handle boundary conditions.
    
    Returns:
    --------
        Tensor
            B-spline basis values, shape (num_splines, num_grid_points + k - 1, num_samples).
    """
    num_splines = tf.shape(grid)[0]
    num_samples = tf.shape(x)[0]

    if extend:
        grid = extend_grid_tf(grid, k)

    x = tf.broadcast_to(x, (num_splines, num_samples))

    # Initialize B_0
    B = tf.cast(tf.logical_and(x >= grid[:, :-1], x < grid[:, 1:]), dtype=tf.float32)

    # Recursive calculation of B_k
    for d in range(1, k):
        left_term = (x - grid[:, :-d-1]) / (grid[:, d:-1] - grid[:, :-d-1])
        right_term = (grid[:, d+1:] - x) / (grid[:, d+1:] - grid[:, 1:-d])
        B = left_term * B[:, :-1] + right_term * B[:, 1:]

    return tf.transpose(B, perm=[0, 2, 1])  # Reshape to (num_splines, num_samples, num_grid_points + k - 1)

class KAN(tf.keras.models.Sequential):
    def __init__(self, layers_configurations, **kwargs):
        super(KAN, self).__init__()
        for layer_config in layers_configurations:
            self.add(KANLinear(**layer_config, **kwargs))

def get_activations(model, model_inputs, layer_name=None):
    layer_outputs = [layer.output for layer in model.layers if layer.name == layer_name or layer_name is None]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(model_inputs)
    return activations

