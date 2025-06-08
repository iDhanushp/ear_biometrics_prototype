#!/usr/bin/env python3
import numpy as np
import librosa
from scipy import signal, stats
from sklearn.preprocessing import StandardScaler
import pywt
from typing import Dict, Tuple
import json
import os
import glob

def extract_ear_canal_features(audio: np.ndarray, sr: int = 44100) -> Dict[str, float]:
    """
    Extract comprehensive features for ear canal biometric authentication.
    Optimized: Reuse STFT and magnitude for spectral features to reduce redundant computation.
    """
    features = {}

    # 1. Enhanced MFCC Features (with derivatives)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    for i in range(mfccs.shape[0]):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        features[f'mfcc_{i+1}_skew'] = stats.skew(mfccs[i])
        features[f'mfcc_{i+1}_kurtosis'] = stats.kurtosis(mfccs[i])
        features[f'mfcc_{i+1}_delta_mean'] = np.mean(mfcc_delta[i])
        features[f'mfcc_{i+1}_delta2_mean'] = np.mean(mfcc_delta2[i])

    # 2. Advanced Spectral Features (reuse STFT/magnitude)
    stft = librosa.stft(audio)
    magnitude = np.abs(stft)
    # Use magnitude for all spectral features that accept S/magnitude
    spectral_centroids = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)[0]
    spectral_contrast = librosa.feature.spectral_contrast(S=magnitude, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
    features['spectral_flatness_mean'] = np.mean(spectral_flatness)
    features['spectral_flatness_std'] = np.std(spectral_flatness)
    for i in range(spectral_contrast.shape[0]):
        features[f'spectral_contrast_{i+1}_mean'] = np.mean(spectral_contrast[i])
        features[f'spectral_contrast_{i+1}_std'] = np.std(spectral_contrast[i])
    
    # 3. Resonance Frequency Analysis (Key for ear canal biometrics)
    # Find dominant frequencies and their relationships
    freqs, fft_magnitude = signal.periodogram(audio, sr)
    
    # Find peaks in frequency domain
    peaks, properties = signal.find_peaks(fft_magnitude, height=np.max(fft_magnitude)*0.1)
    
    if len(peaks) > 0:
        # Dominant frequencies
        peak_freqs = freqs[peaks]
        peak_magnitudes = fft_magnitude[peaks]
        
        # Sort by magnitude
        sorted_indices = np.argsort(peak_magnitudes)[::-1]
        
        # Top 5 resonant frequencies
        for i in range(min(5, len(sorted_indices))):
            idx = sorted_indices[i]
            features[f'resonant_freq_{i+1}'] = peak_freqs[idx]
            features[f'resonant_magnitude_{i+1}'] = peak_magnitudes[idx]
        
        # Fill missing values if less than 5 peaks
        for i in range(len(sorted_indices), 5):
            features[f'resonant_freq_{i+1}'] = 0
            features[f'resonant_magnitude_{i+1}'] = 0
            
        # Frequency ratios (important for ear canal geometry)
        if len(sorted_indices) >= 2:
            features['freq_ratio_1_2'] = peak_freqs[sorted_indices[0]] / peak_freqs[sorted_indices[1]]
        else:
            features['freq_ratio_1_2'] = 0
            
        if len(sorted_indices) >= 3:
            features['freq_ratio_1_3'] = peak_freqs[sorted_indices[0]] / peak_freqs[sorted_indices[2]]
        else:
            features['freq_ratio_1_3'] = 0
    else:
        # No peaks found
        for i in range(5):
            features[f'resonant_freq_{i+1}'] = 0
            features[f'resonant_magnitude_{i+1}'] = 0
        features['freq_ratio_1_2'] = 0
        features['freq_ratio_1_3'] = 0
    
    # 4. Wavelet Transform Features
    # Use Daubechies wavelets for time-frequency analysis
    coeffs = pywt.wavedec(audio, 'db4', level=5)
    
    for i, coeff in enumerate(coeffs):
        features[f'wavelet_level_{i}_energy'] = np.sum(coeff**2)
        features[f'wavelet_level_{i}_std'] = np.std(coeff)
        features[f'wavelet_level_{i}_mean'] = np.mean(coeff)
    
    # 5. Temporal Features
    # Zero crossing rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
    features['zero_crossing_rate_mean'] = np.mean(zero_crossing_rate)
    features['zero_crossing_rate_std'] = np.std(zero_crossing_rate)
    
    # RMS energy
    rms = librosa.feature.rms(y=audio)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    
    # Tempo and rhythm features
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
    features['tempo'] = tempo
    features['beat_count'] = len(beats)
    
    # 6. Statistical Features of Raw Audio
    features['audio_mean'] = np.mean(audio)
    features['audio_std'] = np.std(audio)
    features['audio_skew'] = stats.skew(audio)
    features['audio_kurtosis'] = stats.kurtosis(audio)
    features['audio_entropy'] = -np.sum(audio**2 * np.log(audio**2 + 1e-10))
    
    # 7. Cepstral Features
    # Linear prediction cepstral coefficients
    lpc_coeffs = librosa.lpc(audio, order=10)
    for i, coeff in enumerate(lpc_coeffs[1:]):  # Skip first coefficient
        features[f'lpc_{i+1}'] = coeff
    
    # 8. Chroma Features (harmonic content)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    for i in range(chroma.shape[0]):
        features[f'chroma_{i+1}_mean'] = np.mean(chroma[i])
        features[f'chroma_{i+1}_std'] = np.std(chroma[i])
    
    # 9. Advanced Voice Features (Formants, Pitch, Jitter, Shimmer, HNR)
    # Only compute if audio is long enough
    if len(audio) > sr // 2:
        # Pitch (F0)
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_values = pitches[magnitudes > np.median(magnitudes)]
            if len(pitch_values) > 0:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
        except Exception:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
        # Formant estimation (simple LPC peak method)
        try:
            from scipy.signal import find_peaks
            lpc_order = 12
            lpc_coeffs = librosa.lpc(audio, order=lpc_order)
            w, h = signal.freqz([1], lpc_coeffs, worN=512, fs=sr)
            peaks, _ = find_peaks(-np.abs(h), distance=sr//2000)
            formants = w[peaks][:4] if len(peaks) >= 4 else np.pad(w[peaks], (0, 4-len(peaks)), 'constant')
            for i, f in enumerate(formants):
                features[f'formant_{i+1}'] = f
        except Exception:
            for i in range(4):
                features[f'formant_{i+1}'] = 0
        # Jitter and shimmer (simple frame-to-frame variation)
        try:
            frame_length = int(0.03 * sr)
            hop_length = int(0.015 * sr)
            frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
            frame_amps = np.max(frames, axis=0) - np.min(frames, axis=0)
            shimmer = np.std(frame_amps) / (np.mean(frame_amps) + 1e-6)
            features['shimmer'] = shimmer
            # Jitter: pitch period variation (approximate)
            zero_crossings = librosa.zero_crossings(audio, pad=False)
            zc_diff = np.diff(np.where(zero_crossings)[0])
            if len(zc_diff) > 1:
                jitter = np.std(zc_diff) / (np.mean(zc_diff) + 1e-6)
            else:
                jitter = 0
            features['jitter'] = jitter
        except Exception:
            features['shimmer'] = 0
            features['jitter'] = 0
        # Harmonics-to-noise ratio (HNR, simple SNR proxy)
        try:
            S, phase = librosa.magphase(librosa.stft(audio))
            harmonic = librosa.effects.harmonic(audio)
            percussive = librosa.effects.percussive(audio)
            hnr = np.mean(np.abs(harmonic)) / (np.mean(np.abs(percussive)) + 1e-6)
            features['hnr'] = hnr
        except Exception:
            features['hnr'] = 0
    else:
        features['pitch_mean'] = 0
        features['pitch_std'] = 0
        for i in range(4):
            features[f'formant_{i+1}'] = 0
        features['shimmer'] = 0
        features['jitter'] = 0
        features['hnr'] = 0
    return features

def extract_multimodal_features(audio: np.ndarray, sr: int = 44100) -> dict:
    """
    Extract and prefix both echo and voice features from the input audio.
    Returns a dictionary with 'echo_' and 'voice_' prefixed features for multi-modal support.
    """
    # --- Echo features ---
    # For echo, use the full signal or a segment (customize as needed)
    echo_features = extract_ear_canal_features(audio, sr)
    echo_features = {f'echo_{k}': v for k, v in echo_features.items()}

    # --- Voice features ---
    # For voice, use MFCCs, spectral, and temporal features (customize as needed)
    # Here, we use the same function, but in practice you may want to use a different extractor
    voice_features = extract_ear_canal_features(audio, sr)
    voice_features = {f'voice_{k}': v for k, v in voice_features.items()}

    # Combine
    features = {}
    features.update(echo_features)
    features.update(voice_features)
    return features

def compute_liveness_score(in_ear_audio, open_air_audio, sr=44100):
    """
    Compute a simple liveness score between in-ear and open-air samples.
    Example: RMS energy difference (can be replaced with more advanced metric).
    """
    rms_in_ear = np.mean(librosa.feature.rms(y=in_ear_audio))
    rms_open_air = np.mean(librosa.feature.rms(y=open_air_audio))
    # Score: difference normalized by in-ear energy
    score = (rms_in_ear - rms_open_air) / (rms_in_ear + 1e-6)
    return score

def extract_dataset_features(recordings_dir: str) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Extract enhanced features from all recordings in the dataset.
    Recursively searches all subdirectories for .wav files.
    Returns:
        features_matrix: N x M matrix where N is samples and M is features
        labels: User labels
        feature_names: List of feature names
    """
    import os
    import pandas as pd
    from tqdm import tqdm
    
    features_list = []
    labels = []
    # Recursively find all .wav files and their meta
    wav_files = []
    for root, dirs, files in os.walk(recordings_dir):
        for f in files:
            if f.endswith('.wav') and '_openair' not in f:
                wav_files.append(os.path.join(root, f))
    print(f"Extracting enhanced features from {len(wav_files)} in-ear recordings (excluding open-air)...")
    for audio_path in tqdm(wav_files):
        # Get user ID and meta path
        fname = os.path.basename(audio_path)
        user_id = fname.split('_')[1]
        meta_path = audio_path.replace('.wav', '_meta.json')
        # Find corresponding open-air sample
        openair_path = audio_path.replace('.wav', '_openair.wav')
        openair_meta_path = audio_path.replace('.wav', '_openair_meta.json')
        # Load in-ear audio
        audio, _ = librosa.load(audio_path, sr=44100)
        # Extract multi-modal features
        features = extract_multimodal_features(audio, sr=44100, meta_path=meta_path)
        # Try to load open-air sample for liveness
        if os.path.exists(openair_path):
            openair_audio, _ = librosa.load(openair_path, sr=44100)
            liveness_score = compute_liveness_score(audio, openair_audio, sr=44100)
        else:
            liveness_score = 0.0  # or np.nan
        features['liveness_score'] = liveness_score
        features_list.append(features)
        labels.append(user_id)
    # Convert to DataFrame for easier handling
    features_df = pd.DataFrame(features_list)
    feature_names = list(features_df.columns)
    features_df = features_df.fillna(0)
    print(f"Extracted {len(feature_names)} features per recording")
    print(f"Total recordings: {len(features_list)}")
    print(f"Users: {len(set(labels))}")
    return features_df.values, np.array(labels), feature_names

def extract_multimodal_features(audio: np.ndarray, sr: int, meta_path: str = None) -> dict:
    """
    Extract and separate echo-specific and voice-specific features for multi-modal fusion.
    Args:
        audio (np.ndarray): Audio signal
        sr (int): Sample rate
        meta_path (str, optional): Path to metadata JSON file
    Returns:
        dict: Combined feature dictionary with 'echo_' and 'voice_' prefixes
    """
    # Default: assume both modalities unless specified
    modality = None
    if meta_path and os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            # Heuristic: if 'phrase' in meta, treat as voice; else echo
            if 'phrase' in meta and meta['phrase']:
                modality = 'voice'
            else:
                modality = 'echo'
    # If no meta, fallback to filename heuristic
    elif meta_path:
        if 'phrase' in meta_path:
            modality = 'voice'
        else:
            modality = 'echo'
    # Extract all features
    all_features = extract_ear_canal_features(audio, sr)
    echo_features = {}
    voice_features = {}
    # Assign features to branches (simple heuristic: resonance for echo, MFCC/LPC/chroma for voice)
    for k, v in all_features.items():
        if k.startswith('resonant_') or k.startswith('freq_ratio') or k.startswith('wavelet_'):
            echo_features['echo_' + k] = v
        elif k.startswith('mfcc_') or k.startswith('lpc_') or k.startswith('chroma_'):
            voice_features['voice_' + k] = v
        else:
            # Shared or ambiguous features (e.g., spectral, temporal, stats)
            echo_features['echo_' + k] = v
            voice_features['voice_' + k] = v
    # Only keep the relevant branch if modality is known
    if modality == 'echo':
        return echo_features
    elif modality == 'voice':
        return voice_features
    else:
        # If unknown, return both merged
        combined = {}
        combined.update(echo_features)
        combined.update(voice_features)
        return combined

if __name__ == "__main__":
    # Test feature extraction
    X, y, feature_names = extract_dataset_features('recordings')
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of features: {len(feature_names)}")
    print("\nFirst 10 features:")
    for i, name in enumerate(feature_names[:10]):
        print(f"{i+1}. {name}")