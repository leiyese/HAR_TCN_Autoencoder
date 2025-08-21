"""1D Convolutional Autoencoder for UCI HAR Inertial Signals

This module provides a 1D convolutional autoencoder designed for temporal sensor data.
It's optimized for the UCI HAR dataset with input shape (128, 9) representing
128 timesteps across 9 sensor channels.

The autoencoder learns to compress temporal patterns and reconstruct the input,
making it suitable for anomaly detection by comparing reconstruction error.
"""

from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
import numpy as np


def build_conv1d_autoencoder(
    input_shape: Tuple[int, int] = (128, 9),
    latent_dim: int = 32,
    filters: Tuple[int, ...] = (32, 64, 128),
    kernel_size: int = 3,
    dropout_rate: float = 0.2
) -> Tuple[Model, Model, Model]:
    """Build a 1D convolutional autoencoder for temporal sensor data.
    
    Args:
        input_shape: Input shape (timesteps, channels), default (128, 9)
        latent_dim: Size of the compressed latent representation
        filters: Number of filters for each conv layer
        kernel_size: Convolution kernel size
        dropout_rate: Dropout rate for regularization
        
    Returns:
        (autoencoder, encoder, decoder) models
    """
    timesteps, channels = input_shape
    
    # ==================== ENCODER ====================
    encoder_input = layers.Input(shape=input_shape, name="encoder_input")
    x = encoder_input
    
    # Progressive downsampling with Conv1D
    for i, f in enumerate(filters):
        x = layers.Conv1D(f, kernel_size, activation='relu', padding='same', 
                         name=f'enc_conv1d_{i+1}')(x)
        x = layers.BatchNormalization(name=f'enc_bn_{i+1}')(x)
        x = layers.MaxPooling1D(2, padding='same', name=f'enc_pool_{i+1}')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f'enc_dropout_{i+1}')(x)
    
    # Flatten and compress to latent space
    x = layers.Flatten(name='enc_flatten')(x)
    x = layers.Dense(latent_dim * 2, activation='relu', name='enc_dense_1')(x)
    latent = layers.Dense(latent_dim, activation='linear', name='latent')(x)
    
    encoder = Model(encoder_input, latent, name="encoder")
    
    # ==================== DECODER ====================
    # Calculate the shape after encoding for proper reconstruction
    # After 3 pooling operations: 128 -> 64 -> 32 -> 16
    encoded_timesteps = timesteps // (2 ** len(filters))
    encoded_features = filters[-1]
    
    decoder_input = layers.Input(shape=(latent_dim,), name="decoder_input")
    x = layers.Dense(latent_dim * 2, activation='relu', name='dec_dense_1')(decoder_input)
    x = layers.Dense(encoded_timesteps * encoded_features, activation='relu', 
                    name='dec_dense_2')(x)
    x = layers.Reshape((encoded_timesteps, encoded_features), name='dec_reshape')(x)
    
    # Progressive upsampling with Conv1D
    for i, f in enumerate(reversed(filters[:-1])):
        x = layers.UpSampling1D(2, name=f'dec_upsample_{i+1}')(x)
        x = layers.Conv1D(f, kernel_size, activation='relu', padding='same',
                         name=f'dec_conv1d_{i+1}')(x)
        x = layers.BatchNormalization(name=f'dec_bn_{i+1}')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f'dec_dropout_{i+1}')(x)
    
    # Final upsampling and reconstruction
    x = layers.UpSampling1D(2, name='dec_final_upsample')(x)
    
    # Final convolution to get back to original channels
    decoded = layers.Conv1D(channels, kernel_size, activation='linear', 
                           padding='same', name='reconstruction')(x)
    
    decoder = Model(decoder_input, decoded, name="decoder")
    
    # ==================== AUTOENCODER ====================
    autoencoder_output = decoder(encoder(encoder_input))
    autoencoder = Model(encoder_input, autoencoder_output, name="autoencoder")
    
    return autoencoder, encoder, decoder


def compile_autoencoder(model: Model, learning_rate: float = 1e-3) -> None:
    """Compile the autoencoder with appropriate loss and optimizer."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )


def train_autoencoder(
    model: Model,
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    batch_size: int = 64,
    epochs: int = 100,
    patience: int = 15,
    verbose: int = 1
) -> tf.keras.callbacks.History:
    """Train the autoencoder with early stopping.
    
    Args:
        model: Compiled autoencoder model
        X_train: Training data, shape (n_samples, timesteps, channels)
        X_val: Optional validation data
        batch_size: Training batch size
        epochs: Maximum number of epochs
        patience: Early stopping patience
        verbose: Verbosity level
        
    Returns:
        Training history
    """
    # Setup callbacks
    callback_list = [
        callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=patience//2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train the model
    validation_data = (X_val, X_val) if X_val is not None else None
    
    history = model.fit(
        X_train, X_train,  # Autoencoder: input = target
        validation_data=validation_data,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callback_list,
        verbose=verbose
    )
    
    return history


def compute_reconstruction_error(model: Model, X: np.ndarray) -> np.ndarray:
    """Compute reconstruction error for anomaly detection.
    
    Args:
        model: Trained autoencoder
        X: Input data, shape (n_samples, timesteps, channels)
        
    Returns:
        Reconstruction errors, shape (n_samples,)
    """
    X_reconstructed = model.predict(X, verbose=0)
    
    # Compute MSE per sample
    mse_per_sample = np.mean((X - X_reconstructed) ** 2, axis=(1, 2))
    
    return mse_per_sample


def find_anomaly_threshold(
    reconstruction_errors: np.ndarray,
    percentile: float = 95.0
) -> float:
    """Find anomaly threshold based on reconstruction error distribution.
    
    Args:
        reconstruction_errors: Array of reconstruction errors from normal data
        percentile: Percentile to use as threshold (e.g., 95 = top 5% are anomalies)
        
    Returns:
        Threshold value
    """
    return np.percentile(reconstruction_errors, percentile)


def save_autoencoder(model: Model, filepath: str) -> None:
    """Save the autoencoder model."""
    model.save(filepath)
    print(f"Model saved to {filepath}")


def load_autoencoder(filepath: str) -> Model:
    """Load a saved autoencoder model."""
    return tf.keras.models.load_model(filepath)


if __name__ == "__main__":
    # Example usage and testing
    print("Building 1D Convolutional Autoencoder...")
    
    # Build model
    autoencoder, encoder, decoder = build_conv1d_autoencoder(
        input_shape=(128, 9),
        latent_dim=32,
        filters=(32, 64, 128)
    )
    
    # Compile
    compile_autoencoder(autoencoder, learning_rate=1e-3)
    
    # Print model summaries
    print("\n" + "="*50)
    print("AUTOENCODER SUMMARY")
    print("="*50)
    autoencoder.summary()
    
    print("\n" + "="*50)
    print("ENCODER SUMMARY")  
    print("="*50)
    encoder.summary()
    
    print("\n" + "="*50)
    print("DECODER SUMMARY")
    print("="*50)
    decoder.summary()
    
    # Test with dummy data
    print("\n" + "="*50)
    print("TESTING WITH DUMMY DATA")
    print("="*50)
    
    dummy_data = np.random.randn(10, 128, 9).astype(np.float32)
    print(f"Input shape: {dummy_data.shape}")
    
    # Test encoding
    encoded = encoder.predict(dummy_data, verbose=0)
    print(f"Encoded shape: {encoded.shape}")
    
    # Test reconstruction
    reconstructed = autoencoder.predict(dummy_data, verbose=0)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Test reconstruction error
    errors = compute_reconstruction_error(autoencoder, dummy_data)
    print(f"Reconstruction errors shape: {errors.shape}")
    print(f"Mean reconstruction error: {errors.mean():.6f}")
    
    print("\nModel is ready for training!")
