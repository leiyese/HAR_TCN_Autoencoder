"""
Train the autoencoder model
"""
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_har_data
from src.autoencoder_model import build_autoencoder
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Output directory for reports/plots
report_dir = os.path.join('reports', 'autoencoder', datetime.now().strftime('%Y%m%d-%H%M%S'))
os.makedirs(report_dir, exist_ok=True)

# Load data
train_df, test_df, activity_labels = load_har_data()
X_train = train_df.iloc[:, 2:].values  # Skip Subject and Activity columns
X_test = test_df.iloc[:, 2:].values

# Build model
input_dim = X_train.shape[1]
autoencoder, encoder = build_autoencoder(input_dim)
autoencoder.compile(optimizer='adam', loss='mse')

# Train with callbacks to mitigate overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)

history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(X_test, X_test),
    callbacks=[early_stop, reduce_lr]
)

# Save model
os.makedirs('models', exist_ok=True)
autoencoder.save('models/autoencoder.h5')
print("Autoencoder training complete. Model saved to models/autoencoder.h5")

# Plot training curves (loss)
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Loss')
plt.ylabel('MSE Loss')
plt.xlabel('Epoch')
plt.legend()
ae_curve_path = os.path.join(report_dir, 'autoencoder_loss_curves.png')
plt.tight_layout()
plt.savefig(ae_curve_path)
plt.close()

# Reconstruction error analysis
train_recon = autoencoder.predict(X_train, verbose=0)
test_recon = autoencoder.predict(X_test, verbose=0)

# Use MSE per sample as reconstruction error
train_err = np.mean((X_train - train_recon) ** 2, axis=1)
test_err = np.mean((X_test - test_recon) ** 2, axis=1)

# Threshold at 95th percentile of train errors
threshold = np.percentile(train_err, 95)

# Plot histogram of reconstruction errors with threshold
plt.figure(figsize=(9, 5))
sns.histplot(train_err, bins=50, color='blue', label='Train', stat='density', alpha=0.5)
sns.histplot(test_err, bins=50, color='orange', label='Test', stat='density', alpha=0.5)
plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold (95th)\n{threshold:.4e}')
plt.title('Reconstruction Error Distribution (MSE)')
plt.xlabel('Reconstruction Error')
plt.ylabel('Density')
plt.legend()
hist_path = os.path.join(report_dir, 'reconstruction_error_hist.png')
plt.tight_layout()
plt.savefig(hist_path)
plt.close()

# Binary evaluation: Normal (SITTING/STANDING/LAYING) vs Anomaly (others)
static_ids = {k for k, v in activity_labels.items() if v.upper() in {"SITTING", "STANDING", "LAYING"}}
y_test_true_bin = np.array([1 if a in static_ids else 0 for a in test_df['Activity'].values])  # 1=Normal, 0=Anomaly
y_test_pred_bin = (test_err <= threshold).astype(int)  # 1=Normal, 0=Anomaly

cm_bin = confusion_matrix(y_test_true_bin, y_test_pred_bin, labels=[1, 0])
report_bin = classification_report(y_test_true_bin, y_test_pred_bin, target_names=['Anomaly(0)', 'Normal(1)'], digits=4)

# Save binary classification report
with open(os.path.join(report_dir, 'autoencoder_binary_report.txt'), 'w') as f:
    f.write(f"Threshold: {threshold:.6e}\n\n")
    f.write(report_bin)

# Plot binary confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Normal(1)', 'Anomaly(0)'],
            yticklabels=['Normal(1)', 'Anomaly(0)'])
plt.title('Autoencoder Binary Confusion Matrix (Test)')
plt.xlabel('Predicted')
plt.ylabel('True')
cm_bin_path = os.path.join(report_dir, 'autoencoder_binary_confusion_matrix.png')
plt.tight_layout()
plt.savefig(cm_bin_path)
plt.close()

# Save arrays
np.save(os.path.join(report_dir, 'train_recon_error.npy'), train_err)
np.save(os.path.join(report_dir, 'test_recon_error.npy'), test_err)
np.save(os.path.join(report_dir, 'cm_binary.npy'), cm_bin)

# Normalized binary confusion matrix (row-normalized)
with np.errstate(invalid='ignore', divide='ignore'):
    cm_bin_norm = cm_bin.astype(float) / cm_bin.sum(axis=1, keepdims=True)
cm_bin_norm = np.nan_to_num(cm_bin_norm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_bin_norm, annot=True, fmt='.2f', cmap='Greens',
            xticklabels=['Normal(1)', 'Anomaly(0)'],
            yticklabels=['Normal(1)', 'Anomaly(0)'], vmin=0.0, vmax=1.0)
plt.title('Autoencoder Binary Confusion Matrix (Normalized, Test)')
plt.xlabel('Predicted')
plt.ylabel('True')
cm_bin_norm_path = os.path.join(report_dir, 'autoencoder_binary_confusion_matrix_normalized.png')
plt.tight_layout()
plt.savefig(cm_bin_norm_path)
plt.close()

np.save(os.path.join(report_dir, 'cm_binary_normalized.npy'), cm_bin_norm)

print(f"Training curves saved to {ae_curve_path}")
print(f"Reconstruction error histogram saved to {hist_path}")
print(f"Binary confusion matrix saved to {cm_bin_path}")
print(f"Normalized binary confusion matrix saved to {cm_bin_norm_path}")
