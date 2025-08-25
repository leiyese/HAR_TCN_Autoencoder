"""
Run inference through the full pipeline
"""
import argparse
import numpy as np
import tensorflow as tf
from src.data_loader import load_har_data

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--sample_id', type=int, required=True,
                   help='ID of sample to test (row index)')
args = parser.parse_args()

# Load models
autoencoder = tf.keras.models.load_model('models/autoencoder.h5')
tcn = tf.keras.models.load_model('models/tcn_classifier.h5')

# Load data
_, test_df, activity_labels = load_har_data()

# Get sample
sample = test_df.iloc[args.sample_id, 2:].values  # Skip Subject and Activity
sample = np.expand_dims(sample, axis=0)

# Run through autoencoder
reconstructed = autoencoder.predict(sample)

# Run through TCN
tcn_input = np.expand_dims(sample, axis=1)  # Add timestep dimension
prediction = tcn.predict(tcn_input)
predicted_class = np.argmax(prediction) + 1  # Convert back to 1-based index

# Print results
print(f"Sample {args.sample_id} results:")
print(f"Predicted activity: {activity_labels[predicted_class]}")
print(f"Confidence: {np.max(prediction):.2%}")
