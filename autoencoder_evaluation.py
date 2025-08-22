"""Autoencoder Evaluation and Anomaly Detection for HAR

This module provides comprehensive evaluation of the trained autoencoder
and demonstrates its use for anomaly detection in the HAR pipeline.

Key Features:
- Load trained autoencoder and evaluate performance
- Test anomaly detection on all activity types
- Provide metrics and visualizations
- Demonstrate integration with TCN pipeline
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Tuple, Optional
import logging
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import data_loader
from autoencoder_model import compute_reconstruction_error, get_anomaly_threshold

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoencoderEvaluator:
    """Comprehensive evaluation of trained autoencoder for anomaly detection."""

    def __init__(self, model_path: str, config_path: str, scaler_path: str):
        """
        Initialize evaluator with trained model artifacts.

        Args:
            model_path: Path to trained autoencoder model
            config_path: Path to training configuration and metrics
            scaler_path: Path to fitted data scaler
        """
        self.model_path = model_path
        self.config_path = config_path
        self.scaler_path = scaler_path

        # Load artifacts
        self.model = None
        self.config = None
        self.scaler = None
        self.threshold = None

        self._load_artifacts()
        logger.info("AutoencoderEvaluator initialized")

    def _load_artifacts(self):
        """Load trained model, configuration, and preprocessing artifacts."""
        # Load model with custom_objects to handle metric issues
        import tensorflow as tf

        try:
            # First try loading normally
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Model loaded successfully from: {self.model_path}")
        except (ValueError, TypeError) as e:
            if "Could not deserialize" in str(e) or "mse" in str(e):
                logger.warning(f"Model loading failed due to metric serialization: {e}")
                logger.info("Attempting to load model without compilation...")

                try:
                    # Try loading model without compilation
                    self.model = tf.keras.models.load_model(
                        self.model_path,
                        compile=False,  # Skip compilation to avoid metric issues
                    )

                    # Recompile the model with simple metrics
                    self.model.compile(
                        optimizer="adam",
                        loss="mse",
                        metrics=["mae"],  # Use simple metrics that are always available
                    )
                    logger.info("Model loaded and recompiled successfully")

                except Exception as e2:
                    logger.warning(f"Failed to load model file: {e2}")
                    logger.info(
                        "Attempting to reconstruct model from config and weights..."
                    )

                    # Last resort: reconstruct model from config
                    self._reconstruct_model_from_config()

            else:
                raise e

        # Load configuration
        with open(self.config_path, "r") as f:
            self.config = json.load(f)
        self.threshold = self.config["anomaly_threshold"]
        logger.info(f"Configuration loaded. Threshold: {self.threshold:.6f}")

        # Load scaler
        import joblib

        self.scaler = joblib.load(self.scaler_path)
        logger.info(f"Scaler loaded from: {self.scaler_path}")

    def _reconstruct_model_from_config(self):
        """Reconstruct model from configuration and load weights."""
        logger.info("Reconstructing model from configuration...")

        # Load config first to get model parameters
        with open(self.config_path, "r") as f:
            config = json.load(f)

        model_config = config["model_config"]

        # Import and create model
        from autoencoder_model import create_autoencoder

        autoencoder = create_autoencoder(**model_config)
        self.model = autoencoder.get_model()

        # Try to load weights
        weights_path = self.model_path.replace(".h5", "_weights.h5").replace(
            "autoencoder_", "autoencoder_weights_"
        )
        if Path(weights_path).exists():
            self.model.load_weights(weights_path)
            logger.info(f"Model weights loaded from: {weights_path}")
        else:
            logger.warning(
                "No weights file found. Using random weights - predictions will be meaningless!"
            )

        logger.info("Model reconstructed successfully")

    def load_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Load and preprocess test data for evaluation.

        Returns:
            Tuple of (X_test_norm, y_test, activity_labels, activity_names)
        """
        logger.info("Loading test data for evaluation...")

        # Load raw test data
        _, X_test, _, y_test, _, _, activity_labels = (
            data_loader.load_har_inertial_data()
        )

        # Apply same preprocessing as training
        X_test_norm = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1]))
        X_test_norm = X_test_norm.reshape(X_test.shape)

        logger.info(f"Test data loaded: {X_test_norm.shape}")
        logger.info(f"Activities: {list(activity_labels.values())}")

        return X_test_norm, y_test, activity_labels

    def evaluate_reconstruction_quality(
        self, X_test: np.ndarray, y_test: np.ndarray, activity_labels: Dict
    ) -> Dict[str, float]:
        """
        Evaluate reconstruction quality across different activities.

        Args:
            X_test: Test data
            y_test: Test labels
            activity_labels: Activity label mapping

        Returns:
            Dictionary with reconstruction metrics by activity
        """
        logger.info("Evaluating reconstruction quality...")

        # Compute reconstruction errors
        reconstruction_errors = compute_reconstruction_error(
            self.model, X_test, metric="mse"
        )

        # Group by activity
        results = {}
        for activity_id, activity_name in activity_labels.items():
            mask = y_test == activity_id
            if mask.sum() > 0:
                activity_errors = reconstruction_errors[mask]
                results[activity_name] = {
                    "mean_error": float(np.mean(activity_errors)),
                    "std_error": float(np.std(activity_errors)),
                    "median_error": float(np.median(activity_errors)),
                    "max_error": float(np.max(activity_errors)),
                    "min_error": float(np.min(activity_errors)),
                    "sample_count": int(mask.sum()),
                }

        # Overall statistics
        results["OVERALL"] = {
            "mean_error": float(np.mean(reconstruction_errors)),
            "std_error": float(np.std(reconstruction_errors)),
            "threshold": float(self.threshold),
        }

        logger.info("Reconstruction quality evaluation completed")
        return results

    def evaluate_anomaly_detection(
        self, X_test: np.ndarray, y_test: np.ndarray, activity_labels: Dict
    ) -> Dict[str, any]:
        """
        Evaluate anomaly detection performance.

        Args:
            X_test: Test data
            y_test: Test labels
            activity_labels: Activity label mapping

        Returns:
            Dictionary with anomaly detection metrics
        """
        logger.info("Evaluating anomaly detection performance...")

        # Compute reconstruction errors
        reconstruction_errors = compute_reconstruction_error(
            self.model, X_test, metric="mse"
        )

        # Define ground truth: Static activities (4,5,6) = Normal, Dynamic (1,2,3) = Anomaly
        static_activities = [4, 5, 6]  # SITTING, STANDING, LAYING
        dynamic_activities = [1, 2, 3]  # WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS

        # Create binary labels: 0 = Normal (static), 1 = Anomaly (dynamic)
        y_true = np.array([1 if label in dynamic_activities else 0 for label in y_test])
        y_pred = (reconstruction_errors > self.threshold).astype(int)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, reconstruction_errors)
        roc_auc = auc(fpr, tpr)

        results = {
            "confusion_matrix": cm.tolist(),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "specificity": float(specificity),
            "roc_auc": float(roc_auc),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "threshold": float(self.threshold),
        }

        logger.info(
            f"Anomaly detection results: Accuracy={accuracy:.3f}, F1={f1:.3f}, AUC={roc_auc:.3f}"
        )
        return results

    def plot_reconstruction_errors_by_activity(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        activity_labels: Dict,
        save_path: Optional[str] = None,
    ):
        """Plot reconstruction errors grouped by activity."""
        reconstruction_errors = compute_reconstruction_error(
            self.model, X_test, metric="mse"
        )

        # Prepare data for plotting
        plot_data = []
        for i, (error, label) in enumerate(zip(reconstruction_errors, y_test)):
            plot_data.append(
                {
                    "Reconstruction Error": error,
                    "Activity": activity_labels[label],
                    "Activity Type": "Static" if label in [4, 5, 6] else "Dynamic",
                }
            )

        df = pd.DataFrame(plot_data)

        # Create plot
        plt.figure(figsize=(14, 8))

        # Box plot
        plt.subplot(2, 2, 1)
        sns.boxplot(data=df, x="Activity", y="Reconstruction Error")
        plt.axhline(
            y=self.threshold,
            color="red",
            linestyle="--",
            label=f"Threshold: {self.threshold:.4f}",
        )
        plt.xticks(rotation=45)
        plt.title("Reconstruction Error by Activity")
        plt.legend()

        # Violin plot
        plt.subplot(2, 2, 2)
        sns.violinplot(data=df, x="Activity Type", y="Reconstruction Error")
        plt.axhline(
            y=self.threshold,
            color="red",
            linestyle="--",
            label=f"Threshold: {self.threshold:.4f}",
        )
        plt.title("Reconstruction Error: Static vs Dynamic")
        plt.legend()

        # Histogram
        plt.subplot(2, 2, 3)
        static_errors = df[df["Activity Type"] == "Static"]["Reconstruction Error"]
        dynamic_errors = df[df["Activity Type"] == "Dynamic"]["Reconstruction Error"]

        plt.hist(
            static_errors, bins=50, alpha=0.7, label="Static Activities", density=True
        )
        plt.hist(
            dynamic_errors, bins=50, alpha=0.7, label="Dynamic Activities", density=True
        )
        plt.axvline(
            x=self.threshold,
            color="red",
            linestyle="--",
            label=f"Threshold: {self.threshold:.4f}",
        )
        plt.xlabel("Reconstruction Error")
        plt.ylabel("Density")
        plt.title("Distribution of Reconstruction Errors")
        plt.legend()

        # ROC curve
        plt.subplot(2, 2, 4)
        y_true = np.array([1 if label in [1, 2, 3] else 0 for label in y_test])
        fpr, tpr, _ = roc_curve(y_true, reconstruction_errors)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve for Anomaly Detection")
        plt.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Reconstruction error plots saved to: {save_path}")

        plt.show()

    def demonstrate_pipeline_integration(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        activity_labels: Dict,
        n_samples: int = 10,
    ):
        """
        Demonstrate how the autoencoder integrates with the full HAR pipeline.
        """
        print("=== HAR Pipeline Integration Demo ===\n")

        # Sample some test data
        indices = np.random.choice(len(X_test), n_samples, replace=False)
        sample_data = X_test[indices]
        sample_labels = y_test[indices]

        # Stage 1: Autoencoder Anomaly Detection
        reconstruction_errors = compute_reconstruction_error(
            self.model, sample_data, metric="mse"
        )
        anomaly_predictions = reconstruction_errors > self.threshold

        print("Stage 1: Autoencoder Anomaly Detection")
        print("-" * 50)

        for i, (idx, error, is_anomaly, true_label) in enumerate(
            zip(indices, reconstruction_errors, anomaly_predictions, sample_labels)
        ):
            true_activity = activity_labels[true_label]
            predicted_type = "ANOMALY (Dynamic)" if is_anomaly else "NORMAL (Static)"
            correct = (
                "✅"
                if (is_anomaly and true_label in [1, 2, 3])
                or (not is_anomaly and true_label in [4, 5, 6])
                else "❌"
            )

            print(
                f"Sample {i+1:2d}: Error={error:.6f} → {predicted_type:20s} | True: {true_activity:20s} {correct}"
            )

        print(f"\nThreshold: {self.threshold:.6f}")
        print(
            f"Accuracy: {np.mean([(pred and true in [1,2,3]) or (not pred and true in [4,5,6]) for pred, true in zip(anomaly_predictions, sample_labels)]):.3f}"
        )

        # Stage 2: Simulated TCN Classification (for demonstration)
        print(f"\nStage 2: TCN Classification (Simulated)")
        print("-" * 50)

        for i, (is_anomaly, true_label) in enumerate(
            zip(anomaly_predictions, sample_labels)
        ):
            if is_anomaly:
                # Simulate TCN classification for dynamic activities
                if true_label == 1:
                    tcn_prediction = "WALKING"
                elif true_label == 2:
                    tcn_prediction = "WALKING_UPSTAIRS"
                elif true_label == 3:
                    tcn_prediction = "WALKING_DOWNSTAIRS"
                else:
                    tcn_prediction = "UNKNOWN_DYNAMIC"

                final_prediction = tcn_prediction
            else:
                # Static activity classification
                final_prediction = "STATIC_ACTIVITY"

            true_activity = activity_labels[true_label]
            print(
                f"Sample {i+1:2d}: Final Prediction: {final_prediction:20s} | True: {true_activity}"
            )

    def run_complete_evaluation(
        self, save_dir: str = "evaluation_results"
    ) -> Dict[str, any]:
        """
        Run complete evaluation pipeline.

        Args:
            save_dir: Directory to save evaluation results

        Returns:
            Dictionary with all evaluation results
        """
        logger.info("=== Starting Complete Autoencoder Evaluation ===")

        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        # Load test data
        X_test, y_test, activity_labels = self.load_test_data()

        # Evaluate reconstruction quality
        reconstruction_results = self.evaluate_reconstruction_quality(
            X_test, y_test, activity_labels
        )

        # Evaluate anomaly detection
        anomaly_results = self.evaluate_anomaly_detection(
            X_test, y_test, activity_labels
        )

        # Create visualizations
        plot_path = save_path / "reconstruction_analysis.png"
        self.plot_reconstruction_errors_by_activity(
            X_test, y_test, activity_labels, str(plot_path)
        )

        # Demonstrate pipeline integration
        self.demonstrate_pipeline_integration(X_test, y_test, activity_labels)

        # Save results
        results = {
            "reconstruction_quality": reconstruction_results,
            "anomaly_detection": anomaly_results,
            "model_config": self.config,
            "evaluation_timestamp": pd.Timestamp.now().isoformat(),
        }

        results_path = save_path / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Evaluation results saved to: {results_path}")
        logger.info("=== Evaluation Completed ===")

        return results


def main():
    """Main evaluation function."""
    # Example usage - adjust paths to your trained model
    model_path = (
        "models/autoencoder/autoencoder_20250822_095830.h5"  # Update with actual path
    )
    config_path = (
        "models/autoencoder/config_20250822_095830.json"  # Update with actual path
    )
    scaler_path = (
        "models/autoencoder/scaler_20250822_095830.pkl"  # Update with actual path
    )

    try:
        evaluator = AutoencoderEvaluator(model_path, config_path, scaler_path)
        results = evaluator.run_complete_evaluation()

        print("\n=== Key Results ===")
        print(
            f"Anomaly Detection Accuracy: {results['anomaly_detection']['accuracy']:.3f}"
        )
        print(f"F1 Score: {results['anomaly_detection']['f1_score']:.3f}")
        print(f"ROC AUC: {results['anomaly_detection']['roc_auc']:.3f}")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(
            "\nPlease update the file paths in main() to point to your trained model files."
        )
        print(
            "You can find these files in the models/autoencoder/ directory after training."
        )


if __name__ == "__main__":
    main()
