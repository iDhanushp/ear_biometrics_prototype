#!/usr/bin/env python3
import os
import json
import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def extract_features(audio_path, sr=44100):
    """Extract audio features from a recording.
    
    Args:
        audio_path (str): Path to the audio file
        sr (int): Sample rate
    
    Returns:
        dict: Dictionary of extracted features
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Extract features
    features = {}
    
    # MFCCs (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # Flatten MFCC features into individual columns
    for i in range(mfccs.shape[0]):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
    
    # Zero crossing rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    features['zero_crossing_rate_mean'] = np.mean(zero_crossing_rate)
    features['zero_crossing_rate_std'] = np.std(zero_crossing_rate)
    
    # Root mean square energy
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    
    return features

def load_dataset(recordings_dir):
    """Load all recordings and extract features.
    
    Args:
        recordings_dir (str): Path to recordings directory
    
    Returns:
        tuple: (X, y) where X is feature matrix and y is user labels
    """
    features_list = []
    labels = []
    
    # Get all wav files
    wav_files = [f for f in os.listdir(recordings_dir) if f.endswith('.wav')]
    
    print("Extracting features from recordings...")
    for wav_file in tqdm(wav_files):
        # Get user ID from filename
        user_id = wav_file.split('_')[1]
        
        # Extract features
        audio_path = os.path.join(recordings_dir, wav_file)
        features = extract_features(audio_path)
        
        features_list.append(features)
        labels.append(user_id)
    
    # Convert to DataFrame
    X = pd.DataFrame(features_list)
    y = np.array(labels)
    
    return X, y

def train_and_evaluate(X, y):
    """Train and evaluate the model.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (np.array): User labels
    
    Returns:
        tuple: (model, scaler, test_accuracy)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save y_test and y_pred for later analysis
    np.save('y_test.npy', y_test)
    np.save('y_pred.npy', y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Feature importance plot
    plt.figure(figsize=(12, 6))
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(data=importances.head(10), x='importance', y='feature')
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return model, scaler, model.score(X_test_scaled, y_test)

def main():
    # Load and process data
    X, y = load_dataset('recordings')
    
    # Train and evaluate model
    model, scaler, accuracy = train_and_evaluate(X, y)
    print(f"\nTest Accuracy: {accuracy:.3f}")
    
    # Save model and scaler
    import joblib
    joblib.dump(model, 'user_identification_model.joblib')
    joblib.dump(scaler, 'feature_scaler.joblib')
    print("\nModel and scaler saved to disk.")

if __name__ == "__main__":
    main() 