"""Autoencoder Training Script for HAR Anomaly Detection

This module provides a complete training pipeline for the 1D convolutional autoencoder
used in the HAR anomaly detection system. The autoencoder is trained exclusively on
static activities (SITTING, STANDING, LAYING) to learn normal patterns and detect
dynamic activities as anomalies.

Key Features:
- Loads and preprocesses inertial sensor data
- Trains autoencoder on static activities only
- Evaluates reconstruction performance
- Saves trained model and anomaly detection threshold
- Provides visualization of training progress and results
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, Optional
import logging
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import data_loader
from autoencoder_model import (
    create_autoencoder,
    compute_reconstruction_error,
    get_anomaly_threshold,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf

tf.random.set_seed(42)


class AutoencoderTrainer:
    """Complete training pipeline for HAR autoencoder."""

    def __init__(
        self,
        model_config: Dict[str, Any] = None,
        training_config: Dict[str, Any] = None,
        save_dir: str = "models",
    ):
        """
        Initialize the autoencoder trainer.

        Args:
            model_config: Configuration for autoencoder model
            training_config: Configuration for training process
            save_dir: Directory to save models and results
        """
        # Default model configuration
        self.model_config = model_config or {
            "input_shape": (128, 9),
            "latent_dim": 32,
            "filters": (16, 32, 64),
            "kernel_size": 3,
            "activation": "relu",
            "dropout_rate": 0.2,
        }

        # Default training configuration
        self.training_config = training_config or {
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001,
            "validation_split": 0.2,
            "early_stopping_patience": 15,
            "reduce_lr_patience": 10,
            "min_lr": 1e-7,
        }

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Initialize components
        self.autoencoder = None
        self.model = None
        self.scaler = None
        self.threshold = None
        self.history = None

        logger.info("AutoencoderTrainer initialized")
        logger.info(f"Model config: {self.model_config}")
        logger.info(f"Training config: {self.training_config}")

    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load HAR data and prepare training dataset with static activities only.

        Returns:
            Tuple of (X_train_static, X_test_static) normalized data
        """
        logger.info("Loading HAR inertial data...")

        # Load raw data
        X_train, X_test, y_train, y_test, subj_train, subj_test, activity_labels = (
            data_loader.load_har_inertial_data()
        )

        logger.info(f"Loaded data - Train: {X_train.shape}, Test: {X_test.shape}")

        # Filter for static activities only (SITTING=4, STANDING=5, LAYING=6)
        static_labels = [4, 5, 6]
        X_train_static, y_train_static, _ = data_loader.filter_static_activities(
            X_train, y_train, subj_train, static_labels
        )
        X_test_static, y_test_static, _ = data_loader.filter_static_activities(
            X_test, y_test, subj_test, static_labels
        )

        logger.info(
            f"Static activities - Train: {X_train_static.shape}, Test: {X_test_static.shape}"
        )
        logger.info(
            f"Static activity distribution - Train: {np.unique(y_train_static, return_counts=True)}"
        )

        # Normalize data
        X_train_norm, X_test_norm, self.scaler = data_loader.preprocess_inertial_data(
            X_train_static, X_test_static
        )

        logger.info("Data preprocessing completed")
        logger.info(
            f"Normalized data range: [{X_train_norm.min():.3f}, {X_train_norm.max():.3f}]"
        )

        return X_train_norm, X_test_norm

    def build_model(self):
        """Build and compile the autoencoder model."""
        logger.info("Building autoencoder model...")

        self.autoencoder = create_autoencoder(**self.model_config)
        self.model = self.autoencoder.get_model()

        # Recompile with training configuration
        self.autoencoder.compile_model(
            learning_rate=self.training_config["learning_rate"]
        )

        logger.info("Model built and compiled successfully")
        logger.info(f"Model parameters: {self.model.count_params():,}")

    def setup_callbacks(self):
        """Setup training callbacks."""
        callbacks = []

        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.training_config["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        )
        callbacks.append(early_stopping)

        # Learning rate reduction
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=self.training_config["reduce_lr_patience"],
            min_lr=self.training_config["min_lr"],
            verbose=1,
        )
        callbacks.append(reduce_lr)

        # Model checkpoint
        checkpoint_path = self.save_dir / "best_autoencoder.h5"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1
        )
        callbacks.append(checkpoint)

        return callbacks

    def train(self, X_train: np.ndarray, X_test: np.ndarray) -> Dict[str, Any]:
        """
        Train the autoencoder model.

        Args:
            X_train: Training data
            X_test: Test data for validation

        Returns:
            Training history dictionary
        """
        logger.info("Starting autoencoder training...")

        callbacks = self.setup_callbacks()

        # Train the model (autoencoder predicts its own input)
        self.history = self.model.fit(
            X_train,
            X_train,  # Input and target are the same
            batch_size=self.training_config["batch_size"],
            epochs=self.training_config["epochs"],
            validation_data=(X_test, X_test),
            callbacks=callbacks,
            verbose=1,
        )

        logger.info("Training completed")

        return self.history.history

    def evaluate_and_set_threshold(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance and set anomaly detection threshold.

        Args:
            X_train: Training data (static activities)
            X_test: Test data (static activities)

        Returns:
            Dictionary with evaluation metrics and threshold
        """
        logger.info("Evaluating model and setting anomaly threshold...")

        # Compute reconstruction errors on training data (static activities)
        train_errors = compute_reconstruction_error(self.model, X_train, metric="mse")
        test_errors = compute_reconstruction_error(self.model, X_test, metric="mse")

        # Set threshold based on training data (95th percentile)
        self.threshold = get_anomaly_threshold(
            train_errors, method="percentile", percentile=95.0
        )

        # Calculate metrics
        metrics = {
            "train_mse_mean": float(np.mean(train_errors)),
            "train_mse_std": float(np.std(train_errors)),
            "test_mse_mean": float(np.mean(test_errors)),
            "test_mse_std": float(np.std(test_errors)),
            "anomaly_threshold": float(self.threshold),
            "train_anomaly_rate": float(np.mean(train_errors > self.threshold)),
            "test_anomaly_rate": float(np.mean(test_errors > self.threshold)),
        }

        logger.info(f"Evaluation metrics: {metrics}")
        logger.info(f"Anomaly threshold set to: {self.threshold:.6f}")

        return metrics

    def save_model_and_artifacts(self, metrics: Dict[str, float]):
        """Save trained model and associated artifacts."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save model - use SavedModel format for better compatibility
        model_path = self.save_dir / f"autoencoder_{timestamp}"
        try:
            # Try saving in SavedModel format first (more robust)
            self.model.save(model_path, save_format="tf")
            logger.info(f"Model saved in SavedModel format to: {model_path}")
        except Exception as e:
            logger.warning(f"SavedModel format failed: {e}")
            # Fallback to H5 format without custom objects
            model_path = self.save_dir / f"autoencoder_{timestamp}.h5"
            self.model.save(model_path, save_traces=False)
            logger.info(f"Model saved in H5 format to: {model_path}")

        # Also save just the weights for maximum compatibility
        weights_path = self.save_dir / f"autoencoder_weights_{timestamp}.h5"
        self.model.save_weights(weights_path)
        logger.info(f"Model weights saved to: {weights_path}")

        # Save scaler
        scaler_path = self.save_dir / f"scaler_{timestamp}.pkl"
        import joblib

        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler saved to: {scaler_path}")

        # Save configuration and metrics
        config_data = {
            "model_config": self.model_config,
            "training_config": self.training_config,
            "metrics": metrics,
            "timestamp": timestamp,
            "anomaly_threshold": float(self.threshold),
        }

        config_path = self.save_dir / f"config_{timestamp}.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"Configuration saved to: {config_path}")

        return {
            "model_path": str(model_path),
            "weights_path": str(weights_path),
            "scaler_path": str(scaler_path),
            "config_path": str(config_path),
        }

    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        if self.history is None:
            logger.warning("No training history to plot")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        axes[0].plot(self.history.history["loss"], label="Training Loss", linewidth=2)
        axes[0].plot(
            self.history.history["val_loss"], label="Validation Loss", linewidth=2
        )
        axes[0].set_title("Model Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # MAE plot
        if "mae" in self.history.history:
            axes[1].plot(self.history.history["mae"], label="Training MAE", linewidth=2)
            axes[1].plot(
                self.history.history["val_mae"], label="Validation MAE", linewidth=2
            )
            axes[1].set_title("Model MAE")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("MAE")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Training history plot saved to: {save_path}")

        plt.show()

    def plot_reconstruction_examples(
        self, X_data: np.ndarray, n_examples: int = 3, save_path: Optional[str] = None
    ):
        """Plot reconstruction examples."""
        if self.model is None:
            logger.warning("No trained model available for reconstruction")
            return

        # Get reconstructions
        reconstructions = self.model.predict(X_data[:n_examples], verbose=0)

        fig, axes = plt.subplots(n_examples, 2, figsize=(12, 3 * n_examples))
        if n_examples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_examples):
            # Original
            for j in range(3):  # First 3 channels (total acceleration)
                axes[i, 0].plot(X_data[i, :, j], label=f"Channel {j+1}", alpha=0.8)
            axes[i, 0].set_title(f"Original Signal {i+1}")
            axes[i, 0].set_xlabel("Timestep")
            axes[i, 0].set_ylabel("Value")
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)

            # Reconstructed
            for j in range(3):  # First 3 channels
                axes[i, 1].plot(
                    reconstructions[i, :, j], label=f"Channel {j+1}", alpha=0.8
                )
            axes[i, 1].set_title(f"Reconstructed Signal {i+1}")
            axes[i, 1].set_xlabel("Timestep")
            axes[i, 1].set_ylabel("Value")
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Reconstruction examples plot saved to: {save_path}")

        plt.show()

    def run_complete_training(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline.

        Returns:
            Dictionary with training results and saved file paths
        """
        logger.info("=== Starting Complete Autoencoder Training Pipeline ===")

        # 1. Load and prepare data
        X_train, X_test = self.load_and_prepare_data()

        # 2. Build model
        self.build_model()

        # 3. Train model
        history = self.train(X_train, X_test)

        # 4. Evaluate and set threshold
        metrics = self.evaluate_and_set_threshold(X_train, X_test)

        # 5. Save everything
        file_paths = self.save_model_and_artifacts(metrics)

        # 6. Create visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Plot training history
        history_plot_path = self.save_dir / f"training_history_{timestamp}.png"
        self.plot_training_history(str(history_plot_path))

        # Plot reconstruction examples
        reconstruction_plot_path = self.save_dir / f"reconstructions_{timestamp}.png"
        self.plot_reconstruction_examples(
            X_test, n_examples=3, save_path=str(reconstruction_plot_path)
        )

        logger.info("=== Training Pipeline Completed Successfully ===")

        return {
            "metrics": metrics,
            "file_paths": file_paths,
            "history": history,
            "plot_paths": {
                "training_history": str(history_plot_path),
                "reconstructions": str(reconstruction_plot_path),
            },
        }


def main():
    """Main training function."""

    # Custom configuration (optional)
    model_config = {
        "input_shape": (128, 9),
        "latent_dim": 32,
        "filters": (16, 32, 64),
        "kernel_size": 3,
        "activation": "relu",
        "dropout_rate": 0.2,
    }

    training_config = {
        "batch_size": 32,
        "epochs": 50,  # Reduced for demo, increase for production
        "learning_rate": 0.001,
        "validation_split": 0.2,
        "early_stopping_patience": 15,
        "reduce_lr_patience": 10,
        "min_lr": 1e-7,
    }

    # Initialize trainer
    trainer = AutoencoderTrainer(
        model_config=model_config,
        training_config=training_config,
        save_dir="models/autoencoder",
    )

    # Run complete training
    results = trainer.run_complete_training()

    print("\n=== Training Results ===")
    print(f"Final validation loss: {results['history']['val_loss'][-1]:.6f}")
    print(f"Anomaly threshold: {results['metrics']['anomaly_threshold']:.6f}")
    print(f"Model saved to: {results['file_paths']['model_path']}")

    return results


if __name__ == "__main__":
    results = main()
