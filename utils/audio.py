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

def record_echo(tone, duration=1.0, output_device=None):
    """Record the echo response from ear canal.
    
    Args:
        tone (np.ndarray): Test tone to play
        duration (float): Recording duration in seconds
        output_device (int, optional): Output device index (if None, use system default)
        
    Returns:
        np.ndarray: Recorded audio
    """
    # If output_device is None, use system default for both playback and recording
    device = output_device if output_device is not None else None
    # Determine output channels for the selected device (if specified)
    output_channels = CHANNELS
    if device is not None:
        device_info = sd.query_devices(device)
        output_channels = device_info['max_output_channels']
        # If device supports stereo, duplicate the tone to both channels
        if output_channels == 2:
            tone = np.column_stack([tone, tone])
    # Play the tone first, blocking, so user can hear it
    sd.play(tone, samplerate=SAMPLE_RATE, blocking=True, device=device)
    # Short pause to allow the echo to develop (optional, can be tuned)
    sd.sleep(50)  # 50 ms pause
    
    # Record the echo (input device is usually mono, but can be set similarly if needed)
    echo = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocking=True,
        device=device
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

def save_recording_to_path(audio_data, wav_path, meta_path, metadata=None):
    """Save audio and metadata to explicit file paths.
    
    Args:
        audio_data (np.ndarray): Audio data to save
        wav_path (str or Path): Full path for WAV file
        meta_path (str or Path): Full path for metadata JSON
        metadata (dict, optional): Metadata to save
    """
    # Ensure parent directories exist
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    
    # Save audio as WAV
    with wave.open(str(wav_path), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
    
    # Save metadata
    with open(meta_path, 'w') as f:
        json.dump(metadata or {}, f, indent=2)

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