#!/usr/bin/env python3
import argparse
import time
from tqdm import tqdm
import numpy as np
from utils import (
    record_echo,
    save_recording,
    save_recording_to_path,  # <-- import the new function
    list_audio_devices
)
import os
from datetime import datetime

# List of test phrases
TEST_PHRASES = [
    "The quick brown fox jumps",
    "How much wood would a woodchuck",
    "Peter Piper picked a peck",
    "She sells seashells by the seashore",
    "Unique New York",
    "Red leather, yellow leather",
    "The early bird catches the worm",
    "A proper copper coffee pot",
    "Six slippery snails slid slowly seaward",
    "How can a clam cram in a clean cream can"
]

def collect_phrase_samples(user_id: str, num_samples: int = 3, record_duration: float = 3.0):
    """Collect voice samples using predefined phrases.
    
    Args:
        user_id (str): User identifier
        num_samples (int): Number of samples per phrase
        record_duration (float): Duration of recording in seconds
    """
    # Ensure voice subfolder exists
    voice_dir = os.path.join('recordings', 'voice')
    os.makedirs(voice_dir, exist_ok=True)
    
    print(f"\nCollecting {num_samples} samples per phrase for user {user_id}")
    print("Please wear your headphones and stay in a quiet environment.")
    print("You will be prompted to read each phrase.")
    print("Press Enter when ready...")
    input()
    
    for phrase in TEST_PHRASES:
        print(f"\nPhrase: '{phrase}'")
        print("Please read this phrase when prompted.")
        print("Press Enter when ready to start recording this phrase...")
        input()
        
        for i in tqdm(range(num_samples), desc=f"Recording '{phrase[:20]}...'"):
            print(f"\nSample {i+1}/{num_samples}")
            print("Recording in 3...")
            time.sleep(1)
            print("2...")
            time.sleep(1)
            print("1...")
            time.sleep(1)
            print("READ NOW!")
            
            # Record voice
            recording = record_echo(np.zeros(1), duration=record_duration)  # Empty tone, just record voice
            
            # Save with phrase info
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Use phrase snippet for filename clarity
            phrase_snippet = phrase[:15].replace(' ', '_').replace(',', '')
            base_filename = f"user_{user_id}_phrase_{phrase_snippet}_{timestamp}_{i+1}"
            voice_dir = os.path.join('recordings', 'voice')
            wav_path = os.path.normpath(os.path.join(voice_dir, base_filename + ".wav"))
            meta_path = os.path.normpath(os.path.join(voice_dir, base_filename + "_meta.json"))
            metadata = {
                "user_id": user_id,
                "phrase": phrase,
                "sample_number": i + 1,
                "timestamp": timestamp
            }
            save_recording_to_path(recording, wav_path, meta_path, metadata=metadata)
            
            if i < num_samples - 1:
                print("Waiting 2 seconds before next sample...")
                time.sleep(2)
        
        print(f"\nCompleted {num_samples} samples for phrase: '{phrase}'")
        print("Take a short break if needed.")
        print("Press Enter when ready for next phrase...")
        input()

def main():
    parser = argparse.ArgumentParser(description="Collect voice samples using predefined phrases")
    parser.add_argument("--user_id", help="User identifier")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples per phrase")
    parser.add_argument("--record_duration", type=float, default=3.0, help="Duration of recording in seconds")
    parser.add_argument("--list_devices", action="store_true", help="List available audio devices")
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        return
    
    if not args.user_id:
        parser.error("--user_id is required unless --list_devices is used")
    
    collect_phrase_samples(
        user_id=args.user_id,
        num_samples=args.samples,
        record_duration=args.record_duration
    )

if __name__ == "__main__":
    main()