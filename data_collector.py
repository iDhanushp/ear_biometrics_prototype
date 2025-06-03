#!/usr/bin/env python3
import argparse
import time
from tqdm import tqdm
import numpy as np
from utils import (
    generate_test_tone,
    record_echo,
    save_recording,
    list_audio_devices
)

def collect_samples(user_id: str, num_samples: int = 20, tone_duration: float = 0.1,
                   record_duration: float = 1.0, freq: int = 1000):
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
    
    # Collect samples
    for i in tqdm(range(num_samples), desc="Recording samples"):
        print(f"\nSample {i+1}/{num_samples}")
        print("Playing tone in 3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        
        # Record echo
        echo = record_echo(tone, duration=record_duration)
        
        # Save recording
        save_recording(echo, user_id)
        
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
        freq=args.freq
    )

if __name__ == "__main__":
    main() 