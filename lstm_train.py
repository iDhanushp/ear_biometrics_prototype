import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping

# Directories
SPEC_ECHO_DIR = os.path.join('spectrograms', 'echo')
SPEC_VOICE_DIR = os.path.join('spectrograms', 'voice')

# Choose which modality to use (voice, echo, or fused)
MODALITY = 'voice'  # Change as needed
SPEC_DIR = SPEC_VOICE_DIR if MODALITY == 'voice' else SPEC_ECHO_DIR

# Load data
X = []
y = []
for fname in os.listdir(SPEC_DIR):
    if not fname.endswith('.npy'):
        continue
    arr = np.load(os.path.join(SPEC_DIR, fname))
    X.append(arr)
    parts = fname.split('_')
    user = parts[1] if len(parts) > 1 else 'unknown'
    y.append(user)

X = np.array(X)
y = np.array(y)

# Normalize X to [0, 1]
X_min = X.min()
X_max = X.max()
X = (X - X_min) / (X_max - X_min)

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
num_classes = len(le.classes_)

# Reshape for LSTM: (samples, timesteps, features)
# We'll treat the time axis as timesteps, and the freq axis as features
# X shape: (N, freq, time) -> (N, time, freq)
X_lstm = np.transpose(X, (0, 2, 1))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# Model definition
model = models.Sequential([
    Input(shape=X_train.shape[1:]),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

# Training
history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# Evaluation
loss, acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {acc:.4f}")

# Save model
model.save(f'lstm_{MODALITY}_model.keras')
print(f"Model saved as lstm_{MODALITY}_model.keras (Keras v3 format)")
