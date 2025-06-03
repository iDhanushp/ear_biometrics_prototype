import numpy as np
import librosa
from scipy import signal
from typing import Dict, Tuple

def extract_mfcc(audio: np.ndarray, sr: int = 44100, n_mfcc: int = 13) -> np.ndarray:
    """Extract MFCC features from audio signal.
    
    Args:
        audio (np.ndarray): Audio signal
        sr (int): Sample rate
        n_mfcc (int): Number of MFCC coefficients
        
    Returns:
        np.ndarray: MFCC features
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # Transpose to get time as first dimension

def extract_spectral_features(audio: np.ndarray, sr: int = 44100) -> Dict[str, np.ndarray]:
    """Extract spectral features from audio signal.
    
    Args:
        audio (np.ndarray): Audio signal
        sr (int): Sample rate
        
    Returns:
        Dict[str, np.ndarray]: Dictionary of spectral features
    """
    # Compute spectrogram
    S = np.abs(librosa.stft(audio))
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    
    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)[0]
    
    return {
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_rolloff': spectral_rolloff,
        'spectral_contrast': contrast
    }

def extract_impulse_response(audio: np.ndarray, sr: int = 44100) -> Tuple[np.ndarray, np.ndarray]:
    """Extract impulse response from audio signal.
    
    Args:
        audio (np.ndarray): Audio signal
        sr (int): Sample rate
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Frequency response (freq, magnitude)
    """
    # Compute frequency response
    freqs, response = signal.freqz(audio)
    
    # Convert to Hz
    freqs_hz = freqs * sr / (2 * np.pi)
    
    # Get magnitude response
    magnitude = np.abs(response)
    
    return freqs_hz, magnitude

def extract_all_features(audio: np.ndarray, sr: int = 44100) -> Dict[str, np.ndarray]:
    """Extract all features from audio signal.
    
    Args:
        audio (np.ndarray): Audio signal
        sr (int): Sample rate
        
    Returns:
        Dict[str, np.ndarray]: Dictionary of all features
    """
    features = {}
    
    # MFCC
    features['mfcc'] = extract_mfcc(audio, sr)
    
    # Spectral features
    spectral_features = extract_spectral_features(audio, sr)
    features.update(spectral_features)
    
    # Impulse response
    freqs, magnitude = extract_impulse_response(audio, sr)
    features['freq_response_freqs'] = freqs
    features['freq_response_magnitude'] = magnitude
    
    return features 