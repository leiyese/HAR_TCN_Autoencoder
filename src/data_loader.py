import os
from collections import Counter

import pandas as pd


DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "har_using_smartphones",
    "UCI HAR Dataset",
)


def _read_feature_list(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["index", "feature"])
    return df["feature"].astype(str).tolist()


def _make_unique(names):
    """Return a list where duplicate names are suffixed with _1, _2, ..."""
    counts = Counter(names)
    seen = {}
    out = []
    for name in names:
        if counts[name] > 1:
            seen[name] = seen.get(name, 0) + 1
            out.append(f"{name}_{seen[name]}")
        else:
            out.append(name)
    return out


def load_features():
    """Load feature names and guarantee uniqueness.

    Returns:
        list of feature names (length should be 561)
    """
    features_path = os.path.join(DATASET_PATH, "features.txt")
    raw = _read_feature_list(features_path)
    return _make_unique(raw)


def load_activity_labels():
    labels_path = os.path.join(DATASET_PATH, "activity_labels.txt")
    labels = pd.read_csv(labels_path, sep=r"\s+", header=None, names=["id", "activity"])
    return dict(zip(labels["id"], labels["activity"]))


def load_set(set_type="train"):
    """Load a train/test split and return a DataFrame with Subject and Activity columns.

    Implementation detail: read X without passing column names to avoid pandas duplicate-name checks,
    then assign unique feature names afterwards.
    """
    features = load_features()
    X_path = os.path.join(DATASET_PATH, set_type, f"X_{set_type}.txt")
    y_path = os.path.join(DATASET_PATH, set_type, f"y_{set_type}.txt")
    subj_path = os.path.join(DATASET_PATH, set_type, f"subject_{set_type}.txt")

    # Read features data without names, then assign (use regex sep to avoid FutureWarning)
    X = pd.read_csv(X_path, sep=r"\s+", header=None, engine="python")
    if X.shape[1] != len(features):
        raise ValueError(
            f"Number of columns in X ({X.shape[1]}) does not match number of features ({len(features)})"
        )
    X.columns = features

    y = pd.read_csv(y_path, header=None, names=["Activity"])  # numeric labels
    subj = pd.read_csv(subj_path, header=None, names=["Subject"])  # subject ids

    df = pd.concat([subj, y, X], axis=1)
    return df


def load_har_data():
    """Return (train_df, test_df, activity_labels) where train/test are DataFrames."""
    train_df = load_set("train")
    test_df = load_set("test")
    activity_labels = load_activity_labels()
    return train_df, test_df, activity_labels


if __name__ == "__main__":
    t, s, labels = load_har_data()
    print("Train shape:", t.shape)
    print("Test shape:", s.shape)
    print("Activity labels:", labels)
