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

def collect_samples(user_id: str, num_samples: int = 20, tone_duration: float = 0.1,
                   record_duration: float = 1.0, freq: int = 1000, output_device=None):
    """Collect ear canal echo samples for a user.
    
    Args:
        user_id (str): User identifier
        num_samples (int): Number of samples to collect
        tone_duration (float): Duration of test tone in seconds
        record_duration (float): Duration of echo recording in seconds
        freq (int): Frequency of test tone in Hz
    """
    print(f"\nCollecting {num_samples} samples for user {user_id}")
    print("Please wear your earphones/headset and stay in a quiet environment.")
    print("Press Enter when ready...")
    input()
    
    # Generate test tone
    tone = generate_test_tone(duration=tone_duration, freq=freq)
    
    # Ensure echo subfolder exists
    echo_dir = os.path.join('recordings', 'echo')
    os.makedirs(echo_dir, exist_ok=True)
    
    # Collect samples
    for i in tqdm(range(num_samples), desc="Recording samples"):
        print(f"\nSample {i+1}/{num_samples}")
        print("Playing tone in 3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        # Play a cue beep (optional, 440 Hz, 0.2s)
        cue_beep = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.2, int(44100 * 0.2), False))
        sd.play(cue_beep, samplerate=44100, blocking=True)
        # Now play and record the test tone for echo capture
        echo = record_echo(tone, duration=record_duration, output_device=output_device)
        
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

        # Liveness/occlusion (open-air) sample
        print("\n[OPTIONAL] Liveness/Occlusion Test: Please REMOVE the earbud and press Enter to record open-air sample...")
        input()
        print("Recording open-air (earbud out) sample...")
        openair_echo = record_echo(tone, duration=record_duration, output_device=output_device)
        openair_base_filename = f"user_{user_id}_echo_{timestamp}_{i+1}_openair"
        openair_wav_path = os.path.join(echo_dir, openair_base_filename + ".wav")
        openair_meta_path = os.path.join(echo_dir, openair_base_filename + "_meta.json")
        openair_metadata = metadata.copy()
        openair_metadata["in_ear"] = False
        save_recording_to_path(openair_echo, openair_wav_path, openair_meta_path, metadata=openair_metadata)

        print("You may re-insert the earbud now.")

        # Wait between samples
        if i < num_samples - 1:
            print("Waiting 2 seconds before next sample...")
            time.sleep(2)

def main():
    parser = argparse.ArgumentParser(description="Collect ear canal echo samples for biometric authentication")
    parser.add_argument("--user_id", help="User identifier")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to collect")
    parser.add_argument("--tone_duration", type=float, default=0.1, help="Duration of test tone in seconds")
    parser.add_argument("--record_duration", type=float, default=1.0, help="Duration of echo recording in seconds")
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