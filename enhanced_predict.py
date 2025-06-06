#!/usr/bin/env python3
import numpy as np
import librosa
import joblib
import sys
import pandas as pd
from enhanced_features import extract_multimodal_features

class EnhancedEarCanalPredictor:
    def __init__(self, model_prefix='enhanced_ear_biometric'):
        """Load the enhanced trained model and preprocessing components."""
        try:
            self.model = joblib.load(f'{model_prefix}_model.joblib')
            self.scaler = joblib.load(f'{model_prefix}_scaler.joblib')
            self.label_encoder = joblib.load(f'{model_prefix}_label_encoder.joblib')
            
            # Try to load feature selector
            try:
                self.feature_selector = joblib.load(f'{model_prefix}_feature_selector.joblib')
            except FileNotFoundError:
                self.feature_selector = None
                
            # Load feature names
            try:
                with open(f'{model_prefix}_feature_names.txt', 'r') as f:
                    self.feature_names = [line.strip() for line in f.readlines()]
            except FileNotFoundError:
                self.feature_names = None
                
            print(f"Enhanced model loaded successfully!")
            print(f"Model type: {type(self.model).__name__}")
            print(f"Number of users: {len(self.label_encoder.classes_)}")
            print(f"Users: {', '.join(self.label_encoder.classes_)}")
            
        except Exception as e:
            print(f"Error loading enhanced model: {e}")
            sys.exit(1)
    
    def predict(self, audio_file_path, return_probabilities=False):
        """
        Predict user identity from an audio file.
        
        Args:
            audio_file_path (str): Path to the audio file
            return_probabilities (bool): Whether to return prediction probabilities
            
        Returns:
            str or tuple: Predicted user ID, optionally with probabilities
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_file_path, sr=44100)
            # Find metadata path
            meta_path = audio_file_path.replace('.wav', '_meta.json')
            # Extract multi-modal features
            features = extract_multimodal_features(audio, sr, meta_path)
            # Convert to DataFrame to match training format
            feature_df = pd.DataFrame([features])
            feature_df = feature_df.fillna(0)
            feature_vector = feature_df.values
            # Scale features FIRST (before feature selection)
            feature_vector_scaled = self.scaler.transform(feature_vector)
            # Apply feature selection AFTER scaling
            if self.feature_selector:
                feature_vector_final = self.feature_selector.transform(feature_vector_scaled)
            else:
                feature_vector_final = feature_vector_scaled
            # Make prediction
            prediction = self.model.predict(feature_vector_final)[0]
            predicted_user = self.label_encoder.inverse_transform([prediction])[0]
            if return_probabilities:
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(feature_vector_final)[0]
                    prob_dict = {user: prob for user, prob in zip(self.label_encoder.classes_, probabilities)}
                    return predicted_user, prob_dict
                else:
                    return predicted_user, None
            return predicted_user
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
    
    def predict_with_confidence(self, audio_file_path, confidence_threshold=0.7):
        """
        Predict with confidence score and threshold.
        
        Args:
            audio_file_path (str): Path to the audio file
            confidence_threshold (float): Minimum confidence for acceptance
            
        Returns:
            tuple: (predicted_user, confidence, accepted)
        """
        result = self.predict(audio_file_path, return_probabilities=True)
        
        if result is None:
            return None, 0.0, False
            
        predicted_user, probabilities = result
        
        if probabilities is None:
            return predicted_user, 1.0, True  # Assume high confidence if no proba
        
        confidence = max(probabilities.values())
        accepted = confidence >= confidence_threshold
        
        return predicted_user, confidence, accepted
    
    def analyze_audio_file(self, audio_file_path):
        """
        Provide detailed analysis of an audio file.
        
        Args:
            audio_file_path (str): Path to the audio file
        """
        print(f"\n=== Audio Analysis: {audio_file_path} ===")
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_file_path, sr=44100)
            print(f"Audio duration: {len(audio)/sr:.2f} seconds")
            print(f"Sample rate: {sr} Hz")
            print(f"Audio samples: {len(audio)}")
            
            # Extract features
            features = extract_multimodal_features(audio, sr)
            print(f"Extracted features: {len(features)}")
            
            # Make prediction with confidence
            predicted_user, confidence, accepted = self.predict_with_confidence(
                audio_file_path, confidence_threshold=0.7
            )
            
            print(f"\n=== Prediction Results ===")
            print(f"Predicted User: {predicted_user}")
            print(f"Confidence: {confidence:.3f}")
            print(f"Accepted (>70% confidence): {'Yes' if accepted else 'No'}")
            
            # Get detailed probabilities
            result = self.predict(audio_file_path, return_probabilities=True)
            if result and result[1]:
                _, probabilities = result
                print(f"\n=== All User Probabilities ===")
                sorted_probs = sorted(probabilities.items(), 
                                    key=lambda x: x[1], reverse=True)
                for user, prob in sorted_probs:
                    print(f"{user:15} : {prob:.3f} ({prob*100:.1f}%)")
            
        except Exception as e:
            print(f"Error analyzing audio file: {e}")

def main():
    """Command line interface for enhanced prediction."""
    if len(sys.argv) < 2:
        print("Usage: python enhanced_predict.py <audio_file_path> [--detailed]")
        print("Example: python enhanced_predict.py recordings/user_Abhi_phrase_Hello_20250604_131205.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    detailed = '--detailed' in sys.argv
    
    # Initialize predictor
    predictor = EnhancedEarCanalPredictor()
    
    if detailed:
        # Detailed analysis
        predictor.analyze_audio_file(audio_file)
    else:
        # Quick prediction
        predicted_user, confidence, accepted = predictor.predict_with_confidence(audio_file)
        
        if predicted_user:
            status = "✓ ACCEPTED" if accepted else "✗ REJECTED"
            print(f"Prediction: {predicted_user} (Confidence: {confidence:.3f}) - {status}")
        else:
            print("Prediction failed")

if __name__ == "__main__":
    main()