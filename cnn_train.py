import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Directories
SPEC_ECHO_DIR = os.path.join('spectrograms', 'echo')
SPEC_VOICE_DIR = os.path.join('spectrograms', 'voice')

# Choose which modality to use (voice, echo, or fused)
# Set MODALITY = 'voice', 'echo', or 'fused' (if fused spectrograms are available)
MODALITY = 'voice'  # Change as needed
SPEC_DIR = SPEC_VOICE_DIR if MODALITY == 'voice' else SPEC_ECHO_DIR

# Load data
X = []
y = []
for fname in os.listdir(SPEC_DIR):
    if not fname.endswith('.npy'):
        continue
    arr = np.load(os.path.join(SPEC_DIR, fname))
    # Optionally, resize or pad to fixed shape
    X.append(arr)
    # Extract user label from filename (assumes format: user_<User>_...)
    parts = fname.split('_')
    user = parts[1] if len(parts) > 1 else 'unknown'
    y.append(user)

X = np.array(X)
y = np.array(y)

# Normalize X to [0, 1]
X_min = X.min()
X_max = X.max()
X = (X - X_min) / (X_max - X_min)

# Add channel dimension for CNN
X = X[..., np.newaxis]

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
num_classes = len(le.classes_)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# Data augmentation: random time shift and random noise (train set only)
def augment_spectrograms(X, max_shift=4, noise_std=0.01):
    X_aug = np.empty_like(X)
    for i, x in enumerate(X):
        # Random time shift (axis=1)
        shift = np.random.randint(-max_shift, max_shift + 1)
        x_shifted = np.roll(x, shift, axis=1)
        # Zero-pad the rolled region
        if shift > 0:
            x_shifted[:, :shift, ...] = 0
        elif shift < 0:
            x_shifted[:, shift:, ...] = 0
        # Add random Gaussian noise
        noise = np.random.normal(0, noise_std, size=x_shifted.shape)
        x_noisy = x_shifted + noise
        # Clip to [0, 1]
        x_noisy = np.clip(x_noisy, 0, 1)
        X_aug[i] = x_noisy
    return X_aug

# Augment only the training set
X_train_aug = augment_spectrograms(X_train)

# --- Data/label integrity checks ---
print("Sample filenames and extracted labels:")
for i, fname in enumerate(os.listdir(SPEC_DIR)):
    if i >= 5:
        break
    if fname.endswith('.npy'):
        parts = fname.split('_')
        user = parts[1] if len(parts) > 1 else 'unknown'
        print(f"{fname} -> label: {user}")

print(f"Loaded {len(X)} spectrograms. Shape: {X[0].shape if len(X) > 0 else 'N/A'}")
print(f"Unique labels: {np.unique(y)}")
print("Class distribution:")
from collections import Counter
print(Counter(y))

# --- Visualize a few spectrograms and their labels ---
num_to_plot = min(4, len(X))
plt.figure(figsize=(12, 3 * num_to_plot))
for i in range(num_to_plot):
    plt.subplot(num_to_plot, 1, i + 1)
    plt.imshow(X[i].squeeze(), aspect='auto', origin='lower')
    plt.title(f"Label: {y[i]}")
    plt.colorbar()
plt.tight_layout()
plt.show()

# Model definition
model = models.Sequential([
    Input(shape=X_train.shape[1:]),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

# Early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

# Training
history = model.fit(
    X_train_aug, y_train,
    epochs=40,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# Evaluation
loss, acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {acc:.4f}")

# Save model in new Keras format (Keras 3+)
model.save(f'cnn_{MODALITY}_model.keras')
print(f"Model saved as cnn_{MODALITY}_model.keras (Keras v3 format)")
# Optionally, also save in legacy .h5 for compatibility
# model.save(f'cnn_{MODALITY}_model.h5')
# print(f"Model also saved as cnn_{MODALITY}_model.h5 (legacy HDF5 format)")
