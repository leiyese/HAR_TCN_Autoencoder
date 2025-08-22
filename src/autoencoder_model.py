"""1D Convolutional Autoencoder for HAR Anomaly Detection

This module implements a 1D convolutional autoencoder for human activity recognition.
The model is designed to work with inertial sensor data (128 timesteps × 9 channels)
and is primarily used for anomaly detection on static activities.

Key Features:
- 1D Convolutional layers for temporal feature extraction
- Encoder-decoder architecture with bottleneck
- Reconstruction error computation for anomaly detection
- Compatible with data_loader.py format
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Conv1DAutoencoder:
    """1D Convolutional Autoencoder for temporal sensor data."""

    def __init__(
        self,
        input_shape: Tuple[int, int] = (128, 9),
        latent_dim: int = 32,
        filters: Tuple[int, ...] = (16, 32, 64),
        kernel_size: int = 3,
        activation: str = "relu",
        dropout_rate: float = 0.2,
        name: str = "conv1d_autoencoder",
    ):
        """
        Initialize the 1D Convolutional Autoencoder.

        Args:
            input_shape: Shape of input data (timesteps, channels)
            latent_dim: Dimension of the latent bottleneck
            filters: Tuple of filter sizes for encoder layers
            kernel_size: Convolution kernel size
            activation: Activation function
            dropout_rate: Dropout rate for regularization
            name: Model name
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.name = name

        # Model components
        self.encoder = None
        self.decoder = None
        self.autoencoder = None

        # Build the model
        self._build_model()

    def _build_model(self):
        """Build the complete autoencoder model."""
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.autoencoder = self._build_autoencoder()

        logger.info(f"Built autoencoder with input shape {self.input_shape}")
        logger.info(f"Encoder output shape: {self.encoder.output_shape}")
        logger.info(f"Latent dimension: {self.latent_dim}")

    def _build_encoder(self) -> Model:
        """Build the encoder network."""
        inputs = keras.Input(shape=self.input_shape, name="encoder_input")
        x = inputs

        # Convolutional encoder layers
        for i, filters in enumerate(self.filters):
            x = layers.Conv1D(
                filters=filters,
                kernel_size=self.kernel_size,
                activation=self.activation,
                padding="same",
                name=f"conv1d_enc_{i+1}",
            )(x)
            x = layers.BatchNormalization(name=f"bn_enc_{i+1}")(x)
            x = layers.MaxPooling1D(2, padding="same", name=f"pool_enc_{i+1}")(x)
            x = layers.Dropout(self.dropout_rate, name=f"dropout_enc_{i+1}")(x)

        # Flatten and create bottleneck
        x = layers.Flatten(name="flatten")(x)
        latent = layers.Dense(
            self.latent_dim, activation=self.activation, name="latent"
        )(x)

        encoder = Model(inputs, latent, name="encoder")
        return encoder

    def _build_decoder(self) -> Model:
        """Build the decoder network."""
        # Calculate the shape after encoder convolutions
        # With 3 pooling layers (each divides by 2): 128 -> 64 -> 32 -> 16
        encoded_length = self.input_shape[0]
        for _ in self.filters:
            encoded_length = encoded_length // 2

        final_filters = self.filters[-1]

        latent_inputs = keras.Input(shape=(self.latent_dim,), name="decoder_input")

        # Reshape to feature maps
        x = layers.Dense(
            encoded_length * final_filters,
            activation=self.activation,
            name="dense_reshape",
        )(latent_inputs)
        x = layers.Reshape((encoded_length, final_filters), name="reshape")(x)

        # Transpose convolutional decoder layers (reverse the encoder)
        reversed_filters = list(reversed(self.filters[:-1]))

        for i, filters in enumerate(reversed_filters):
            x = layers.UpSampling1D(2, name=f"upsample_dec_{i+1}")(x)
            x = layers.Conv1DTranspose(
                filters=filters,
                kernel_size=self.kernel_size,
                activation=self.activation,
                padding="same",
                name=f"conv1d_dec_{i+1}",
            )(x)
            x = layers.BatchNormalization(name=f"bn_dec_{i+1}")(x)
            x = layers.Dropout(self.dropout_rate, name=f"dropout_dec_{i+1}")(x)

        # Final upsampling and reconstruction layer
        x = layers.UpSampling1D(2, name="final_upsample")(x)
        outputs = layers.Conv1DTranspose(
            filters=self.input_shape[1],  # Number of input channels
            kernel_size=self.kernel_size,
            activation="linear",  # Linear for reconstruction
            padding="same",
            name="reconstruction",
        )(x)

        # Simple cropping to ensure exact output shape match
        # This handles slight size mismatches due to pooling/upsampling
        if encoded_length * (2 ** len(self.filters)) != self.input_shape[0]:
            # Crop to exact target length
            outputs = layers.Lambda(
                lambda x: x[:, : self.input_shape[0], :], name="crop_to_target_length"
            )(outputs)

        decoder = Model(latent_inputs, outputs, name="decoder")
        return decoder

    def _build_autoencoder(self) -> Model:
        """Build the complete autoencoder by connecting encoder and decoder."""
        inputs = keras.Input(shape=self.input_shape, name="autoencoder_input")
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)

        autoencoder = Model(inputs, decoded, name=self.name)
        return autoencoder

    def compile_model(
        self,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        loss: str = "mse",
        metrics: list = None,
    ):
        """Compile the autoencoder model."""
        if metrics is None:
            metrics = ["mae"]

        # Create optimizer with learning rate
        if optimizer == "adam":
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == "rmsprop":
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = optimizer

        self.autoencoder.compile(optimizer=opt, loss=loss, metrics=metrics)

        logger.info(
            f"Compiled autoencoder with {optimizer} optimizer (lr={learning_rate})"
        )

    def get_model(self) -> Model:
        """Return the compiled autoencoder model."""
        return self.autoencoder

    def get_encoder(self) -> Model:
        """Return the encoder model."""
        return self.encoder

    def get_decoder(self) -> Model:
        """Return the decoder model."""
        return self.decoder

    def summary(self):
        """Print model summaries."""
        print("=== ENCODER ===")
        self.encoder.summary()
        print("\n=== DECODER ===")
        self.decoder.summary()
        print("\n=== AUTOENCODER ===")
        self.autoencoder.summary()


def create_autoencoder(
    input_shape: Tuple[int, int] = (128, 9), latent_dim: int = 32, **kwargs
) -> Conv1DAutoencoder:
    """
    Factory function to create and return a 1D autoencoder instance.

    Args:
        input_shape: Shape of input data (timesteps, channels)
        latent_dim: Dimension of the latent bottleneck
        **kwargs: Additional arguments for Conv1DAutoencoder

    Returns:
        Conv1DAutoencoder: Configured autoencoder instance
    """
    autoencoder = Conv1DAutoencoder(
        input_shape=input_shape, latent_dim=latent_dim, **kwargs
    )

    # Compile with default settings
    autoencoder.compile_model()

    return autoencoder


def compute_reconstruction_error(
    model: Model, data: np.ndarray, metric: str = "mse"
) -> np.ndarray:
    """
    Compute reconstruction error for anomaly detection.

    Args:
        model: Trained autoencoder model
        data: Input data to reconstruct
        metric: Error metric ('mse', 'mae', 'rmse')

    Returns:
        Array of reconstruction errors per sample
    """
    reconstructions = model.predict(data, verbose=0)

    if metric == "mse":
        errors = np.mean((data - reconstructions) ** 2, axis=(1, 2))
    elif metric == "mae":
        errors = np.mean(np.abs(data - reconstructions), axis=(1, 2))
    elif metric == "rmse":
        errors = np.sqrt(np.mean((data - reconstructions) ** 2, axis=(1, 2)))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return errors


def get_anomaly_threshold(
    reconstruction_errors: np.ndarray,
    method: str = "percentile",
    percentile: float = 95.0,
    std_multiplier: float = 2.0,
) -> float:
    """
    Calculate anomaly detection threshold from reconstruction errors.

    Args:
        reconstruction_errors: Array of reconstruction errors
        method: Threshold method ('percentile', 'std', 'iqr')
        percentile: Percentile for percentile method
        std_multiplier: Standard deviation multiplier for std method

    Returns:
        Anomaly threshold value
    """
    if method == "percentile":
        threshold = np.percentile(reconstruction_errors, percentile)
    elif method == "std":
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        threshold = mean_error + std_multiplier * std_error
    elif method == "iqr":
        q75, q25 = np.percentile(reconstruction_errors, [75, 25])
        iqr = q75 - q25
        threshold = q75 + 1.5 * iqr
    else:
        raise ValueError(f"Unknown threshold method: {method}")

    return threshold


if __name__ == "__main__":
    # Test the autoencoder creation
    print("Testing 1D Convolutional Autoencoder...")

    # Create autoencoder
    autoencoder = create_autoencoder(
        input_shape=(128, 9), latent_dim=32, filters=(16, 32, 64), dropout_rate=0.2
    )

    # Print summaries
    autoencoder.summary()

    # Test with dummy data
    dummy_data = np.random.randn(100, 128, 9)
    print(f"\nTesting with dummy data shape: {dummy_data.shape}")

    # Get model and test prediction
    model = autoencoder.get_model()
    predictions = model.predict(dummy_data[:5], verbose=0)
    print(f"Prediction shape: {predictions.shape}")

    # Test reconstruction error computation
    errors = compute_reconstruction_error(model, dummy_data[:10])
    print(f"Reconstruction errors shape: {errors.shape}")
    print(f"Sample errors: {errors[:5]}")

    # Test threshold computation
    threshold = get_anomaly_threshold(errors, method="percentile", percentile=95)
    print(f"Anomaly threshold (95th percentile): {threshold:.4f}")

    print("\n✅ Autoencoder model creation successful!")
