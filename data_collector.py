#!/usr/bin/env python3
import argparse
import time
from tqdm import tqdm
import numpy as np
from utils import (
    generate_test_tone,
    record_echo,
    save_recording,
    save_recording_to_path,  # <-- import the new function
    list_audio_devices
)
import os
from datetime import datetime
import sounddevice as sd  # Add this import for cue beep
import json
from scipy import signal
import librosa

def generate_chirp_tone(duration=0.5, sr=44100, f_start=500, f_end=4000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    chirp = signal.chirp(t, f0=f_start, f1=f_end, t1=duration, method='linear')
    return chirp.astype(np.float32)

def collect_samples(user_id: str, num_samples: int = 20, tone_duration: float = 0.5,
                   record_duration: float = 0.7, freq: int = 1000, output_device=None):
    """Collect ear canal echo samples for a user.
    
    Args:
        user_id (str): User identifier
        num_samples (int): Number of samples to collect
        tone_duration (float): Duration of test tone in seconds
        record_duration (float): Duration of echo recording in seconds
        freq (int): Frequency of test tone in Hz
    """
    print(f"\nCollecting {num_samples} in-ear samples for user {user_id}")
    print("Please wear your earphones/headset and stay in a quiet environment.")
    print("Press Enter when ready...")
    input()
    # Use chirp tone for broader resonance
    tone = generate_chirp_tone(duration=tone_duration)
    
    # Ensure echo subfolder exists
    echo_dir = os.path.join('recordings', 'echo')
    os.makedirs(echo_dir, exist_ok=True)
    
    # Confirm audio output with a test beep
    print("\nPlaying a test beep to confirm audio output. Please listen...")
    cue_beep = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.2, int(44100 * 0.2), False))
    sd.stop()  # Flush any previous playback
    sd.play(cue_beep, samplerate=44100, blocking=True)
    print("If you did not hear the beep, check your audio device and press Enter to try again, or Ctrl+C to abort.")
    input()
    
    # Collect all in-ear samples first
    for i in tqdm(range(num_samples), desc="Recording in-ear samples"):
        print(f"\nSample {i+1}/{num_samples}")
        print("Playing tone in 3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        # Play a cue beep (optional, 440 Hz, 0.2s)
        sd.stop()
        sd.play(cue_beep, samplerate=44100, blocking=True)
        # Now play and record the test tone for echo capture
        time.sleep(0.2)  # Pause before record
        echo = record_echo(tone, duration=record_duration, output_device=output_device)
        rms = float(np.mean(librosa.feature.rms(y=echo)))
        if rms < 0.0005:
            print(f"[Warning] Low RMS ({rms:.6f}) detected. Discarding sample {i+1}. Please adjust earbud and retry.")
            continue
        # Save recording
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"user_{user_id}_echo_{timestamp}_{i+1}"
        wav_path = os.path.join(echo_dir, base_filename + ".wav")
        meta_path = os.path.join(echo_dir, base_filename + "_meta.json")
        metadata = {
            "user_id": user_id,
            "timestamp": timestamp,
            "sample_number": i + 1,
            "tone_duration": tone_duration,
            "record_duration": record_duration,
            "freq": freq,
            "in_ear": True
        }
        save_recording_to_path(echo, wav_path, meta_path, metadata=metadata)

        # Log per-sample RMS and centroid for QA
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=echo, sr=44100)))
        print(f"[QA] Sample {i+1}: RMS={rms:.6f}, Centroid={centroid:.2f}")

        if i < num_samples - 1:
            print("Waiting 2 seconds before next sample...")
            time.sleep(2)
    
    # Now collect all open-air (liveness) samples in a batch
    print("\n[Batch Liveness] Please REMOVE the earbud now. Press Enter when ready to record open-air (liveness) samples...")
    input()
    for i in tqdm(range(num_samples), desc="Recording open-air samples"):
        print(f"\nOpen-air sample {i+1}/{num_samples}")
        print("Playing tone in 3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        # Play a cue beep (optional, 440 Hz, 0.2s)
        sd.stop()
        sd.play(cue_beep, samplerate=44100, blocking=True)
        openair_echo = record_echo(tone, duration=record_duration, output_device=output_device)
        
        # Use the same timestamp and sample number as in-ear for pairing
        # Find the corresponding in-ear meta file
        inear_meta_files = sorted([f for f in os.listdir(echo_dir) if f.startswith(f"user_{user_id}_echo_") and f.endswith(f"_{i+1}_meta.json") and 'openair' not in f])
        if inear_meta_files:
            with open(os.path.join(echo_dir, inear_meta_files[0]), 'r') as f:
                inear_meta = json.load(f)
            timestamp = inear_meta['timestamp']
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        openair_base_filename = f"user_{user_id}_echo_{timestamp}_{i+1}_openair"
        openair_wav_path = os.path.join(echo_dir, openair_base_filename + ".wav")
        openair_meta_path = os.path.join(echo_dir, openair_base_filename + "_meta.json")
        openair_metadata = {
            "user_id": user_id,
            "timestamp": timestamp,
            "sample_number": i + 1,
            "tone_duration": tone_duration,
            "record_duration": record_duration,
            "freq": freq,
            "in_ear": False
        }
        save_recording_to_path(openair_echo, openair_wav_path, openair_meta_path, metadata=openair_metadata)

        if i < num_samples - 1:
            print("Waiting 2 seconds before next open-air sample...")
            time.sleep(2)
    print("\nYou may re-insert the earbud now. Liveness collection complete.")

def main():
    parser = argparse.ArgumentParser(description="Collect ear canal echo samples for biometric authentication")
    parser.add_argument("--user_id", help="User identifier")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to collect")
    parser.add_argument("--tone_duration", type=float, default=0.5, help="Duration of test tone in seconds")
    parser.add_argument("--record_duration", type=float, default=0.7, help="Duration of echo recording in seconds")
    parser.add_argument("--freq", type=int, default=1000, help="Frequency of test tone in Hz")
    parser.add_argument("--list_devices", action="store_true", help="List available audio devices")
    parser.add_argument("--output_device", type=int, default=None, help="Output device index for playback/recording")
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        return
    
    if not args.user_id:
        parser.error("--user_id is required unless --list_devices is used")
    
    collect_samples(
        user_id=args.user_id,
        num_samples=args.samples,
        tone_duration=args.tone_duration,
        record_duration=args.record_duration,
        freq=args.freq,
        output_device=args.output_device
    )

if __name__ == "__main__":
    main()