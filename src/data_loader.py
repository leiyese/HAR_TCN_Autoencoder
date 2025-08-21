"""Data loader for UCI HAR Dataset - Inertial Signals Only

This module loads raw windowed sensor data from the Inertial Signals folder
for use with 1D CNNs, autoencoders, and TCN models.

Data format:
- Input: Raw sensor readings (128 timesteps × 9 channels per window)
- Channels: total_acc_x/y/z, body_acc_x/y/z, body_gyro_x/y/z
- Output shape: (n_windows, 128, 9)
"""

import os
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Dataset path
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "har_using_smartphones",
    "UCI HAR Dataset",
)

# Channel names in order
CHANNEL_NAMES = [
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
]


def load_activity_labels() -> Dict[int, str]:
    """Load activity label mapping."""
    labels_path = os.path.join(DATASET_PATH, "activity_labels.txt")
    labels = pd.read_csv(labels_path, sep=r"\s+", header=None, names=["id", "activity"])
    return dict(zip(labels["id"], labels["activity"]))


def load_inertial_signals(
    split: str = "train",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load raw inertial signals for train or test split.

    Args:
        split: 'train' or 'test'

    Returns:
        X: shape (n_windows, 128, 9) - raw sensor data
        y: shape (n_windows,) - activity labels (1-6)
        subjects: shape (n_windows,) - subject ids (1-30)
    """
    inertial_path = os.path.join(DATASET_PATH, split, "Inertial Signals")

    # Load each channel
    channels = []
    for channel in CHANNEL_NAMES:
        file_path = os.path.join(inertial_path, f"{channel}_{split}.txt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")

        # Load data (n_windows, 128)
        data = np.loadtxt(file_path)
        channels.append(data)

    # Stack channels: (n_windows, 128, 9)
    X = np.stack(channels, axis=-1).astype(np.float32)

    # Load labels and subjects
    y_path = os.path.join(DATASET_PATH, split, f"y_{split}.txt")
    subject_path = os.path.join(DATASET_PATH, split, f"subject_{split}.txt")

    y = np.loadtxt(y_path, dtype=int)
    subjects = np.loadtxt(subject_path, dtype=int)

    # Validate shapes
    assert X.shape[0] == len(y) == len(subjects), "Mismatched sample counts"
    assert X.shape[1] == 128, f"Expected 128 timesteps, got {X.shape[1]}"
    assert X.shape[2] == 9, f"Expected 9 channels, got {X.shape[2]}"

    return X, y, subjects


def load_har_inertial_data():
    """Load complete HAR inertial dataset.

    Returns:
        tuple: (X_train, X_test, y_train, y_test,
                subjects_train, subjects_test, activity_labels)
    """
    X_train, y_train, subjects_train = load_inertial_signals("train")
    X_test, y_test, subjects_test = load_inertial_signals("test")
    activity_labels = load_activity_labels()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        subjects_train,
        subjects_test,
        activity_labels,
    )


def preprocess_inertial_data(
    X_train: np.ndarray, X_test: np.ndarray, normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """Preprocess inertial data for training.

    Args:
        X_train: Training data (n_train, 128, 9)
        X_test: Test data (n_test, 128, 9)
        normalize: Whether to apply standardization

    Returns:
        X_train_processed, X_test_processed, scaler (or None)
    """
    if not normalize:
        return X_train, X_test, None

    # Reshape for scaling: (n_samples * timesteps, channels)
    n_train, timesteps, channels = X_train.shape
    n_test = X_test.shape[0]

    X_train_flat = X_train.reshape(-1, channels)
    X_test_flat = X_test.reshape(-1, channels)

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    # Reshape back
    X_train_processed = X_train_scaled.reshape(n_train, timesteps, channels)
    X_test_processed = X_test_scaled.reshape(n_test, timesteps, channels)

    return X_train_processed, X_test_processed, scaler


def filter_static_activities(X, y, subjects, static_labels=[4, 5, 6]):
    """Filter data to include only static activities (for autoencoder training).

    Args:
        X, y, subjects: Data arrays
        static_labels: Activity labels to keep (default: SITTING=4, STANDING=5, LAYING=6)

    Returns:
        Filtered X, y, subjects
    """
    mask = np.isin(y, static_labels)
    return X[mask], y[mask], subjects[mask]


def get_channel_info():
    """Get information about the 9 sensor channels."""
    return {
        "channels": CHANNEL_NAMES,
        "accelerometer": CHANNEL_NAMES[:6],  # total_acc + body_acc
        "gyroscope": CHANNEL_NAMES[6:],  # body_gyro
        "total_acceleration": CHANNEL_NAMES[:3],
        "body_acceleration": CHANNEL_NAMES[3:6],
        "body_gyroscope": CHANNEL_NAMES[6:],
    }


if __name__ == "__main__":
    # Test the loader
    print("Loading HAR inertial data...")
    X_train, X_test, y_train, y_test, subj_train, subj_test, labels = (
        load_har_inertial_data()
    )

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Activity labels: {labels}")
    print(f"Channels: {CHANNEL_NAMES}")
    print(f"Unique activities in train: {np.unique(y_train)}")
    print(f"Unique subjects in train: {len(np.unique(subj_train))}")

    # Test preprocessing
    X_train_norm, X_test_norm, scaler = preprocess_inertial_data(X_train, X_test)
    print(
        f"After normalization - Train mean: {X_train_norm.mean():.3f}, std: {X_train_norm.std():.3f}"
    )

    # Test static filtering
    X_static, y_static, _ = filter_static_activities(X_train, y_train, subj_train)
    print(f"Static activities shape: {X_static.shape}")
    print(f"Static activity labels: {np.unique(y_static)}")
