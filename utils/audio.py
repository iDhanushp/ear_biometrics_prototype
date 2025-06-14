import numpy as np
import sounddevice as sd
import librosa
import json
from datetime import datetime
from pathlib import Path
import wave
import os
import scipy.io.wavfile

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

def generate_test_beep(duration=1.0, sr=44100, freq=440, amplitude=1.0):
    """Generate a test beep (sine wave) for audibility.
    
    Args:
        duration (float): Duration in seconds
        sr (int): Sample rate in Hz
        freq (float): Frequency in Hz
        amplitude (float): Amplitude (volume) of the beep
        
    Returns:
        np.ndarray: Stereo audio signal
    """
    t = np.linspace(0, duration, int(sr * duration), False)
    beep = amplitude * np.sin(2 * np.pi * freq * t)
    # Stereo
    beep = np.column_stack([beep, beep])
    return beep.astype(np.float32)

def generate_cue_beep(duration=0.7, sr=44100, freq=330, amplitude=1.0):
    """Generate a cue beep (sine wave) for clarity before each sample.
    
    Args:
        duration (float): Duration in seconds
        sr (int): Sample rate in Hz
        freq (float): Frequency in Hz
        amplitude (float): Amplitude (volume) of the beep
        
    Returns:
        np.ndarray: Stereo audio signal
    """
    t = np.linspace(0, duration, int(sr * duration), False)
    beep = amplitude * np.sin(2 * np.pi * freq * t)
    # Stereo
    beep = np.column_stack([beep, beep])
    return beep.astype(np.float32)

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

def robust_play(sound, samplerate=44100, device=None):
    """Play sound on the given device, fallback to default if device fails. Always use stereo if available."""
    try:
        if device is not None:
            device_info = sd.query_devices(device)
            out_channels = device_info['max_output_channels']
            print(f"[DEBUG] robust_play: Using device {device} ({device_info['name']}), output channels: {out_channels}, requested samplerate: {samplerate}")
            print(f"[DEBUG] Device default samplerate: {device_info['default_samplerate']}")
            if out_channels == 2:
                if sound.ndim == 1:
                    sound = np.column_stack([sound, sound])
        else:
            print("[DEBUG] robust_play: Using system default output device")
        sd.stop()
        sd.play(sound, samplerate=samplerate, blocking=True, device=device)
    except Exception as e:
        print(f"[WARN] Playback failed on device {device}: {e}\nFalling back to system default output device.")
        try:
            sd.stop()
            default_info = sd.query_devices(None, 'output')
            out_channels = default_info['max_output_channels']
            print(f"[DEBUG] robust_play: Default device ({default_info['name']}), output channels: {out_channels}, requested samplerate: {samplerate}")
            print(f"[DEBUG] Device default samplerate: {default_info['default_samplerate']}")
            if out_channels == 2 and sound.ndim == 1:
                sound = np.column_stack([sound, sound])
            sd.play(sound, samplerate=samplerate, blocking=True, device=None)
        except Exception as e2:
            print(f"[ERROR] Playback failed on default device: {e2}")
            raise

def validate_echo(signal):
    """Check if echo is valid (not NaN or silent)."""
    if np.isnan(signal).any() or np.isinf(signal).any():
        return False
    if np.max(np.abs(signal)) < 1e-5:
        return False
    return True

def robust_record_echo(tone, duration=1.0, output_device=None, input_device=None):
    device_out = output_device if output_device is not None else None
    device_in = input_device if input_device is not None else None
    info = sd.query_devices(device_in, 'input')
    print(f"[DEBUG] Input Device: {info['name']} | Default Sample Rate: {info['default_samplerate']} | Channels: {info['max_input_channels']}")
    # Always use mono (channel 0)
    channels = 1
    # Use the device's default sample rate if 16000 is not supported
    rec_samplerate = 16000
    if abs(info['default_samplerate'] - 16000) > 1:
        print(f"[WARN] Input device does not support 16kHz, using default sample rate: {info['default_samplerate']}")
        rec_samplerate = int(info['default_samplerate'])
    output_channels = CHANNELS
    if device_out is not None:
        try:
            device_info = sd.query_devices(device_out)
            output_channels = device_info['max_output_channels']
            print(f"[DEBUG] Output Device: {device_info['name']} | Default Sample Rate: {device_info['default_samplerate']} | Channels: {output_channels}")
            if output_channels == 2:
                tone = np.column_stack([tone, tone])
        except Exception as e:
            print(f"[WARN] Output device query failed: {e}\nFalling back to system default.")
            device_out = None
    try:
        sd.play(tone, samplerate=44100, blocking=True, device=device_out)
        sd.sleep(50)
        echo = sd.rec(
            int(duration * rec_samplerate),
            samplerate=rec_samplerate,
            channels=channels,
            dtype=DTYPE,
            blocking=True,
            device=device_in
        )
        echo = echo.flatten()
        if info['max_input_channels'] > 1 and echo.ndim > 1:
            print(f"[DEBUG] Multi-channel input detected, using only channel 0")
            echo = echo[::info['max_input_channels']]
        if not np.all(np.isfinite(echo)):
            print(f"[ERROR] NaN/Inf detected in echo recording!")
            raise ValueError("NaN/Inf in echo recording")
        echo = np.clip(echo, -1.0, 1.0)
        return echo, rec_samplerate
    except Exception as e:
        print(f"[WARN] Playback/record failed on output {device_out} or input {device_in}: {e}\nFalling back to system default.")
        try:
            sd.play(tone, samplerate=44100, blocking=True, device=None)
            sd.sleep(50)
            echo = sd.rec(
                int(duration * rec_samplerate),
                samplerate=rec_samplerate,
                channels=channels,
                dtype=DTYPE,
                blocking=True,
                device=None
            )
            echo = echo.flatten()
            if not np.all(np.isfinite(echo)):
                print(f"[ERROR] NaN/Inf detected in fallback echo recording!")
                raise ValueError("NaN/Inf in fallback echo recording")
            echo = np.clip(echo, -1.0, 1.0)
            return echo, rec_samplerate
        except Exception as e2:
            print(f"[ERROR] Playback/record failed on default device: {e2}")
            raise

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
    clipped = np.clip(audio_data, -1.0, 1.0)
    with wave.open(str(wav_path), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((clipped * 32767).astype(np.int16).tobytes())
    
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

def select_audio_device():
    """Prompt user to select an audio output device interactively."""
    devices = sd.query_devices()
    print("\nAvailable Audio Devices:")
    print("-" * 50)
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']}")
        print(f"   Input channels: {device['max_input_channels']}")
        print(f"   Output channels: {device['max_output_channels']}")
        print("-" * 50)
    while True:
        try:
            idx = input("Enter output device index (or press Enter for default): ")

            if idx == '':
                return None
            idx = int(idx)
            if 0 <= idx < len(devices):
                return idx
            else:
                print("Invalid index. Try again.")
        except Exception:
            print("Invalid input. Enter a valid device index or press Enter.")

def select_audio_input_device():
    """Prompt user to select an audio input device interactively."""
    devices = sd.query_devices()
    print("\nAvailable Audio Input Devices:")
    print("-" * 50)
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"{i}: {device['name']}")
            print(f"   Input channels: {device['max_input_channels']}")
            print(f"   Output channels: {device['max_output_channels']}")
            print("-" * 50)
    while True:
        try:
            idx = input("Enter input device index (or press Enter for default): ").strip()
            if idx == '':
                return None
            idx = int(idx)
            if 0 <= idx < len(devices) and devices[idx]['max_input_channels'] > 0:
                return idx
            else:
                print("Invalid index. Try again.")
        except Exception:
            print("Invalid input. Enter a valid device index or press Enter.")

def interactive_device_test(sound, samplerate=44100):
    """Let the user cycle through all output devices and test playback until they confirm they hear the sound."""
    devices = sd.query_devices()
    output_indices = [i for i, d in enumerate(devices) if d['max_output_channels'] > 0]
    print("\nCycle through available output devices to test playback. Press Enter to skip a device, or 'y' if you hear the beep.")
    for idx in output_indices:
        device = devices[idx]
        print(f"\nTesting device {idx}: {device['name']} (Output channels: {device['max_output_channels']})")
        try:
            robust_play(sound, samplerate=samplerate, device=idx)
        except Exception as e:
            print(f"[ERROR] Playback failed on device {idx}: {e}")
            continue
        resp = input("Did you hear the beep? (y/n, Enter to skip): ").strip().lower()
        if resp == 'y':
            print(f"Selected device {idx}: {device['name']}")
            return idx
    print("No working output device found. Please check your Windows Sound Settings.")
    return None

def test_all_input_devices(duration=2.0, samplerate=44100):
    """Test all available input devices: record and playback a short sample from each."""
    devices = sd.query_devices()
    input_indices = [i for i, d in enumerate(devices) if d['max_input_channels'] > 0]
    print("\nTesting all available input devices. Speak or make a sound near each mic when prompted.")
    for idx in input_indices:
        device = devices[idx]
        print(f"\nTesting input device {idx}: {device['name']} (Input channels: {device['max_input_channels']})")
        try:
            print("Recording...")
            audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.float32, blocking=True, device=idx)
            print("Playing back...")
            sd.play(audio, samplerate=samplerate, blocking=True)
        except Exception as e:
            print(f"[ERROR] Failed on device {idx}: {e}")
        resp = input("Did you hear your recording? (y/n, Enter to continue): ").strip().lower()
        if resp == 'y':
            print(f"Selected input device {idx}: {device['name']}")
            return idx
    print("No working input device selected.")
    return None

def simultaneous_play_and_record(tone, duration=1.0, output_device=None, input_device=None):
    """Simultaneously play tone and record from mic. Useful for HFP-limited TWS setups."""
    # Duplicate tone for stereo if needed
    try:
        device_info = sd.query_devices(output_device)
        if device_info['max_output_channels'] == 2 and tone.ndim == 1:
            tone = np.column_stack([tone, tone])
    except Exception:
        pass
    try:
        sd.play(tone, samplerate=SAMPLE_RATE, device=output_device)
        echo = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            device=input_device,
            blocking=True
        )
        sd.wait()
        echo = echo.flatten()
        if not np.all(np.isfinite(echo)):
            print(f"[ERROR] NaN/Inf detected in simultaneous echo recording!")
            raise ValueError("NaN/Inf in simultaneous echo recording")
        echo = np.clip(echo, -1.0, 1.0)
        return echo
    except Exception as e:
        print(f"[ERROR] Simultaneous play/rec failed: {e}")
        raise