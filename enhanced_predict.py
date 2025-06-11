#!/usr/bin/env python3
import numpy as np
import librosa
import joblib
import sys
import pandas as pd
from enhanced_features import extract_ear_canal_features, extract_multimodal_features

def score_echo_quality(rms, centroid):
    """
    Classify echo signal quality for fallback logic.
    
    Args:
        rms (float): Echo RMS mean
        centroid (float): Echo spectral centroid mean
        
    Returns:
        str: "low" or "usable"
    """
    if rms < 0.0004 or abs(centroid - 700) > 500:
        return "low"
    else:
        return "usable"

class EnhancedEarCanalPredictor:
    def __init__(self, model_prefix='enhanced_ear_biometric', fusion_mode='fused'):
        """Load the enhanced trained model(s) and preprocessing components."""
        self.fusion_mode = fusion_mode
        self.model_prefix = model_prefix
        # Always load fused model for early fusion
        self.model = joblib.load(f'{model_prefix}_model.joblib')
        self.scaler = joblib.load(f'{model_prefix}_scaler.joblib')
        self.label_encoder = joblib.load(f'{model_prefix}_label_encoder.joblib')
        try:
            self.feature_selector = joblib.load(f'{model_prefix}_feature_selector.joblib')
        except FileNotFoundError:
            self.feature_selector = None
        try:
            with open(f'{model_prefix}_feature_names.txt', 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            self.feature_names = None
        # For late fusion, load echo and voice models
        if fusion_mode in ['late', 'hybrid', 'echo', 'voice']:
            self.echo_model = joblib.load('enhanced_ear_biometric_echo_model.joblib')
            self.echo_scaler = joblib.load('enhanced_ear_biometric_echo_scaler.joblib')
            self.echo_selector = joblib.load('enhanced_ear_biometric_echo_feature_selector.joblib')
            self.echo_encoder = joblib.load('enhanced_ear_biometric_echo_label_encoder.joblib')
            with open('enhanced_ear_biometric_echo_feature_names.txt', 'r') as f:
                self.echo_feature_names = [line.strip() for line in f.readlines()]
            self.voice_model = joblib.load('enhanced_ear_biometric_voice_model.joblib')
            self.voice_scaler = joblib.load('enhanced_ear_biometric_voice_scaler.joblib')
            self.voice_selector = joblib.load('enhanced_ear_biometric_voice_feature_selector.joblib')
            self.voice_encoder = joblib.load('enhanced_ear_biometric_voice_label_encoder.joblib')
            with open('enhanced_ear_biometric_voice_feature_names.txt', 'r') as f:
                self.voice_feature_names = [line.strip() for line in f.readlines()]
        print(f"Enhanced model(s) loaded. Fusion mode: {fusion_mode}")

    def predict(self, audio_file_path, return_probabilities=False):
        """
        Predict user identity from an audio file using the selected fusion mode or auto-fallback.
        """
        try:
            audio, sr = librosa.load(audio_file_path, sr=44100)
            features = extract_multimodal_features(audio, sr)
            feature_df = pd.DataFrame([features]).fillna(0)
            feature_vector = feature_df.values
            # --- Echo quality fallback logic ---
            echo_rms = features.get('echo_rms_mean', None)
            echo_centroid = features.get('echo_spectral_centroid_mean', None)
            if echo_rms is not None and echo_centroid is not None and self.fusion_mode == 'fused':
                quality = score_echo_quality(echo_rms, echo_centroid)
                if quality == 'low':
                    # Fallback to voice-only model
                    voice_features = {k: v for k, v in features.items() if k.startswith('voice_')}
                    voice_df = pd.DataFrame([voice_features]).fillna(0)
                    voice_vec = voice_df.values
                    voice_vec_scaled = self.voice_scaler.transform(voice_vec)
                    voice_vec_final = self.voice_selector.transform(voice_vec_scaled)
                    prediction = self.voice_model.predict(voice_vec_final)[0]
                    predicted_user = self.voice_encoder.inverse_transform([prediction])[0]
                    if return_probabilities and hasattr(self.voice_model, 'predict_proba'):
                        probabilities = self.voice_model.predict_proba(voice_vec_final)[0]
                        prob_dict = {user: prob for user, prob in zip(self.voice_encoder.classes_, probabilities)}
                        return predicted_user, prob_dict
                    return predicted_user
                # else: usable, proceed with fused model below
            # --- Fused model (default) ---
            if self.fusion_mode == 'fused':
                feature_vector_scaled = self.scaler.transform(feature_vector)
                if self.feature_selector:
                    feature_vector_final = self.feature_selector.transform(feature_vector_scaled)
                else:
                    feature_vector_final = feature_vector_scaled
                prediction = self.model.predict(feature_vector_final)[0]
                predicted_user = self.label_encoder.inverse_transform([prediction])[0]
                if return_probabilities and hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(feature_vector_final)[0]
                    prob_dict = {user: prob for user, prob in zip(self.label_encoder.classes_, probabilities)}
                    return predicted_user, prob_dict
                return predicted_user
            # Echo-only
            elif self.fusion_mode == 'echo':
                features = extract_multimodal_features(audio, sr)
                echo_features = {k: v for k, v in features.items() if k.startswith('echo_')}
                feature_df = pd.DataFrame([echo_features])
                feature_df = feature_df.fillna(0)
                feature_vector = feature_df.values
                feature_vector_scaled = self.echo_scaler.transform(feature_vector)
                feature_vector_final = self.echo_selector.transform(feature_vector_scaled)
                prediction = self.echo_model.predict(feature_vector_final)[0]
                predicted_user = self.echo_encoder.inverse_transform([prediction])[0]
                if return_probabilities and hasattr(self.echo_model, 'predict_proba'):
                    probabilities = self.echo_model.predict_proba(feature_vector_final)[0]
                    prob_dict = {user: prob for user, prob in zip(self.echo_encoder.classes_, probabilities)}
                    return predicted_user, prob_dict
                return predicted_user
            # Voice-only
            elif self.fusion_mode == 'voice':
                features = extract_multimodal_features(audio, sr)
                voice_features = {k: v for k, v in features.items() if k.startswith('voice_')}
                feature_df = pd.DataFrame([voice_features])
                feature_df = feature_df.fillna(0)
                feature_vector = feature_df.values
                feature_vector_scaled = self.voice_scaler.transform(feature_vector)
                feature_vector_final = self.voice_selector.transform(feature_vector_scaled)
                prediction = self.voice_model.predict(feature_vector_final)[0]
                predicted_user = self.voice_encoder.inverse_transform([prediction])[0]
                if return_probabilities and hasattr(self.voice_model, 'predict_proba'):
                    probabilities = self.voice_model.predict_proba(feature_vector_final)[0]
                    prob_dict = {user: prob for user, prob in zip(self.voice_encoder.classes_, probabilities)}
                    return predicted_user, prob_dict
                return predicted_user
            # Late fusion (average probabilities)
            elif self.fusion_mode == 'late':
                features = extract_multimodal_features(audio, sr)
                echo_features = {k: v for k, v in features.items() if k.startswith('echo_')}
                voice_features = {k: v for k, v in features.items() if k.startswith('voice_')}
                # Echo
                echo_df = pd.DataFrame([echo_features]).fillna(0)
                echo_vec = echo_df.values
                echo_vec_scaled = self.echo_scaler.transform(echo_vec)
                echo_vec_final = self.echo_selector.transform(echo_vec_scaled)
                echo_probs = self.echo_model.predict_proba(echo_vec_final)[0]
                echo_users = self.echo_encoder.classes_
                # Voice
                voice_df = pd.DataFrame([voice_features]).fillna(0)
                voice_vec = voice_df.values
                voice_vec_scaled = self.voice_scaler.transform(voice_vec)
                voice_vec_final = self.voice_selector.transform(voice_vec_scaled)
                voice_probs = self.voice_model.predict_proba(voice_vec_final)[0]
                voice_users = self.voice_encoder.classes_
                # Align user order (assume same set)
                user_list = sorted(set(echo_users) & set(voice_users))
                echo_prob_dict = {u: p for u, p in zip(echo_users, echo_probs) if u in user_list}
                voice_prob_dict = {u: p for u, p in zip(voice_users, voice_probs) if u in user_list}
                avg_probs = {u: (echo_prob_dict[u] + voice_prob_dict[u]) / 2 for u in user_list}
                predicted_user = max(avg_probs, key=avg_probs.get)
                if return_probabilities:
                    return predicted_user, avg_probs
                return predicted_user
            else:
                raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")
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
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced prediction with multi-modal fusion support.")
    parser.add_argument("audio_file", type=str, help="Path to the audio file")
    parser.add_argument("--fusion_mode", type=str, default="fused", choices=["fused", "late", "echo", "voice"], help="Fusion mode: fused (early), late, echo, or voice")
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")
    args = parser.parse_args()
    predictor = EnhancedEarCanalPredictor(fusion_mode=args.fusion_mode)
    if args.detailed:
        predictor.analyze_audio_file(args.audio_file)
    else:
        predicted_user, confidence, accepted = predictor.predict_with_confidence(args.audio_file)
        if predicted_user:
            status = "✓ ACCEPTED" if accepted else "✗ REJECTED"
            print(f"Prediction: {predicted_user} (Confidence: {confidence:.3f}) - {status}")
        else:
            print("Prediction failed")

if __name__ == "__main__":
    main()