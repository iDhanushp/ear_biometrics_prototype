import numpy as np
import sounddevice as sd
import librosa
import json
from datetime import datetime
from pathlib import Path
import wave
import os

# Audio parameters
SAMPLE_RATE = 44100  # Hz
CHANNELS = 1  # Mono
DTYPE = np.float32
CHUNK_SIZE = 1024

def generate_test_tone(duration=0.1, freq=1000):
    """Generate a test tone (sine wave) for ear canal excitation.
    
    Args:
        duration (float): Duration in seconds
        freq (float): Frequency in Hz
        
    Returns:
        np.ndarray: Audio signal
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    tone = np.sin(2 * np.pi * freq * t)
    # Apply fade in/out to avoid clicks
    fade_duration = int(0.01 * SAMPLE_RATE)
    fade_in = np.linspace(0, 1, fade_duration)
    fade_out = np.linspace(1, 0, fade_duration)
    tone[:fade_duration] *= fade_in
    tone[-fade_duration:] *= fade_out
    return tone

def record_echo(tone, duration=1.0):
    """Record the echo response from ear canal.
    
    Args:
        tone (np.ndarray): Test tone to play
        duration (float): Recording duration in seconds
        
    Returns:
        np.ndarray: Recorded audio
    """
    # Play the tone and record simultaneously
    recording = sd.playrec(
        tone,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocking=True
    )
    
    # Record the echo
    echo = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocking=True
    )
    
    return echo.flatten()

def save_recording(audio_data, user_id, output_dir="recordings", metadata=None):
    """Save recording and metadata.
    
    Args:
        audio_data (np.ndarray): Audio data to save
        user_id (str): User identifier
        output_dir (str): Output directory
        metadata (dict, optional): Additional metadata to save
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"user_{user_id}_{timestamp}"
    
    # Save audio as WAV
    wav_path = output_path / f"{filename}.wav"
    with wave.open(str(wav_path), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
    
    # Save metadata
    base_metadata = {
        "user_id": user_id,
        "timestamp": timestamp,
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "duration": len(audio_data) / SAMPLE_RATE,
        "filename": f"{filename}.wav"
    }
    if metadata is not None:
        base_metadata.update(metadata)
    
    meta_path = output_path / f"{filename}_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(base_metadata, f, indent=2)

def list_audio_devices():
    """List available audio input/output devices."""
    devices = sd.query_devices()
    print("\nAvailable Audio Devices:")
    print("-" * 50)
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']}")
        print(f"   Input channels: {device['max_input_channels']}")
        print(f"   Output channels: {device['max_output_channels']}")
        print("-" * 50) 