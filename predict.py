#!/usr/bin/env python3

import argparse
import os
import librosa
import numpy as np
import pandas as pd
import joblib

# Import the extract_features function from train_model.py (or copy it here if you prefer)
# (Assuming extract_features is defined in train_model.py)
from train_model import extract_features

def predict_user(audio_path, model_path='user_identification_model.joblib', scaler_path='feature_scaler.joblib'):
    """Predict the user from a given audio file using the trained model.
    
    Args:
        audio_path (str): Path to the audio file (WAV) to predict.
        model_path (str): Path to the trained model (joblib file).
        scaler_path (str): Path to the saved scaler (joblib file).
    
    Returns:
        tuple: (predicted_user, confidence) where confidence is the probability of the predicted class.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load the trained model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Extract features from the audio file
    features = extract_features(audio_path)

    # Convert features into a DataFrame (one row)
    X = pd.DataFrame([features])

    # Scale the features using the saved scaler
    X_scaled = scaler.transform(X)

    # Predict the user
    predicted_user = model.predict(X_scaled)[0]

    # Compute confidence (probability) of the predicted class
    proba = model.predict_proba(X_scaled)[0]
    confidence = proba[np.argmax(proba)]

    return predicted_user, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict user from an audio file using the trained model.")
    parser.add_argument("--audio", type=str, required=True, help="Path to the audio file (WAV) to predict.")
    parser.add_argument("--model", type=str, default="user_identification_model.joblib", help="Path to the trained model (joblib file).")
    parser.add_argument("--scaler", type=str, default="feature_scaler.joblib", help="Path to the saved scaler (joblib file).")
    args = parser.parse_args()

    try:
        predicted_user, confidence = predict_user(args.audio, args.model, args.scaler)
        print(f"Predicted User: {predicted_user} (Confidence: {confidence:.3f})")
    except Exception as e:
        print("Error:", e) 