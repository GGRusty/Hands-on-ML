import tensorflow as tf

class LayerNormalization(tf.keras.layers.Layer):
    """
    A custom implementation of Layer Normalization in TensorFlow. Layer Normalization is a type of
    normalization technique like Batch Normalization or Instance Normalization. Unlike other
    normalization techniques which normalize the inputs across the batch dimension, Layer
    Normalization applies normalization on each single sample across the features dimension.

    Attributes:
        eps (float): A small number added for numerical stability during division. This helps avoid
                     division by zero errors when the variance is very small. Default value is 0.001.

    Methods:
        call(X): Normalizes the input X across its last dimension and returns the output.
        compute_output_shape(batch_input_shape): Returns the shape of the output given the input shape.
    """

    def __init__(self, eps=0.001, **kwargs):
        """
        Initializes the LayerNormalization layer.

        Args:
            eps (float, optional): A small number added for numerical stability. Defaults to 0.001.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, batch_input_shape):
        """
        Creates the alpha and beta weights with the shape of the last dimension of the input.

        Args:
            batch_input_shape (TensorShape or tuple): The shape of the input data.
        """
        self.alpha = self.add_weight(
            name="alpha", shape=batch_input_shape[-1:], initializer="ones"
        )
        self.beta = self.add_weight(
            name="beta", shape=batch_input_shape[-1:], initializer="zeros"
        )
        super().build(batch_input_shape)

    def call(self, X):
        """
        Normalizes the input X across its last dimension and returns the output.

        Args:
            X (tf.Tensor): The input data.

        Returns:
            The output of the layer.
        """
        mean, variance = tf.nn.moments(X, axes=-1, keepdims=True)
        return self.alpha * (X - mean) / (tf.sqrt(variance + self.eps)) + self.beta

    def compute_output_shape(self, batch_input_shape):
        """
        Returns the shape of the output given the input shape.

        Args:
            batch_input_shape (TensorShape or tuple): The shape of the input data.

        Returns:
            The shape of the output.
        """
        return batch_input_shape

    def get_config(self):
        """
        Returns the configuration of the layer.

        Returns:
            A dictionary containing the configuration of the layer.
        """
        base_config = super().get_config()
        return {**base_config, "eps": self.eps}