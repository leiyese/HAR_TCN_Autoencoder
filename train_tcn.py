"""
Train the TCN classifier model
"""
import os
import numpy as np
import matplotlib
# Use non-interactive backend to avoid Qt issues (e.g., SIGABRT on savefig)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from src.data_loader import load_har_data
from src.tcn_model import build_tcn
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Output directory for reports/plots
report_dir = os.path.join('reports', 'tcn', datetime.now().strftime('%Y%m%d-%H%M%S'))
os.makedirs(report_dir, exist_ok=True)

# Load data
train_df, test_df, activity_labels = load_har_data()

# Prepare data
X_train = train_df.iloc[:, 2:].values  # Features
X_test = test_df.iloc[:, 2:].values
y_train = to_categorical(train_df['Activity'] - 1)  # Convert to 0-based index
y_test = to_categorical(test_df['Activity'] - 1)

# Reshape for TCN (add timesteps dimension)
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

# Build model
input_shape = (X_train.shape[1], X_train.shape[2])
num_classes = len(activity_labels)
model = build_tcn(input_shape, num_classes)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
print("Training TCN model...")
early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=256,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr]
)

# Save model
os.makedirs('models', exist_ok=True)
model.save('models/tcn_classifier.h5')

# Plot training history
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

# Save plots
plt.tight_layout()
plot_path = os.path.join(report_dir, 'training_curves.png')
plt.savefig(plot_path)
plt.close()

print(f"TCN training complete. Model saved to models/tcn_classifier.h5")
print(f"Training curves saved to {plot_path}")

# Evaluate on test set and create confusion matrix & report
print("Evaluating on test set...")
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

cm = confusion_matrix(y_true, y_pred)
classes = list(range(1, len(cm) + 1))  # 1..num_classes for readability

# Save classification report
report_txt = classification_report(y_true, y_pred, digits=4)
with open(os.path.join(report_dir, 'classification_report.txt'), 'w') as f:
    f.write(report_txt)
print("Classification report saved.")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix (Test)')
plt.xlabel('Predicted label')
plt.ylabel('True label')
cm_path = os.path.join(report_dir, 'confusion_matrix.png')
plt.tight_layout()
plt.savefig(cm_path)
plt.close()

# Save raw arrays for reproducibility
np.save(os.path.join(report_dir, 'y_true.npy'), y_true)
np.save(os.path.join(report_dir, 'y_pred.npy'), y_pred)
np.save(os.path.join(report_dir, 'confusion_matrix.npy'), cm)

# Normalized confusion matrix (row-normalized)
with np.errstate(invalid='ignore', divide='ignore'):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=classes, yticklabels=classes, vmin=0.0, vmax=1.0)
plt.title('Confusion Matrix (Normalized, Test)')
plt.xlabel('Predicted label')
plt.ylabel('True label')
cm_norm_path = os.path.join(report_dir, 'confusion_matrix_normalized.png')
plt.tight_layout()
plt.savefig(cm_norm_path)
plt.close()

np.save(os.path.join(report_dir, 'confusion_matrix_normalized.npy'), cm_norm)

# Print final metrics
print("\nFinal Metrics:")
print(f"- Training Accuracy: {history.history['accuracy'][-1]:.2%}")
print(f"- Validation Accuracy: {history.history['val_accuracy'][-1]:.2%}")
print(f"- Training Loss: {history.history['loss'][-1]:.4f}")
print(f"- Validation Loss: {history.history['val_loss'][-1]:.4f}")
print(f"Confusion matrix saved to {cm_path}")
print(f"Classification report saved to {os.path.join(report_dir, 'classification_report.txt')}")
print(f"Normalized confusion matrix saved to {cm_norm_path}")
