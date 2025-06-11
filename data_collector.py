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

def generate_chirp_tone(duration=0.5, sr=44100, f_start=500, f_end=4000, amplitude=0.8):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    chirp = signal.chirp(t, f0=f_start, f1=f_end, t1=duration, method='linear')
    chirp = chirp * amplitude
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
    
    # Open QA log CSV
    qa_log_path = os.path.join(echo_dir, f"qa_log_{user_id}.csv")
    with open(qa_log_path, "a") as qa_log:
        qa_log.write("user_id,timestamp,sample_number,rms,centroid,snr,liveness_score\n")
    
    # Collect all in-ear samples first
    sample_idx = 0
    while sample_idx < num_samples:
        attempts = 0
        while attempts < 3:
            print(f"\nSample {sample_idx+1}/{num_samples} (Attempt {attempts+1}/3)")
            print("Playing tone in 3...")
            time.sleep(1)
            print("2...")
            time.sleep(1)
            print("1...")
            time.sleep(1)
            # Play a cue beep (optional, 440 Hz, 0.2s)
            sd.stop()
            sd.play(cue_beep, samplerate=44100, blocking=True)
            time.sleep(0.2)  # Pause before record
            # Play and record echo (simultaneous play/rec assumed)
            echo = record_echo(tone, duration=record_duration, output_device=output_device)
            rms = float(np.mean(librosa.feature.rms(y=echo)))
            centroid = float(np.mean(librosa.feature.spectral_centroid(y=echo, sr=44100)))
            # Compute simple SNR (signal vs. first 50 ms)
            noise_win = echo[:int(0.05 * 44100)]
            signal_win = echo[int(0.05 * 44100):]
            snr = 10 * np.log10(np.var(signal_win) / (np.var(noise_win) + 1e-9))
            # Liveness score (simple heuristic)
            liveness_score = (rms - 0.0005) * 1000 - abs(centroid - 700)/10
            # Save QA log
            with open(qa_log_path, "a") as qa_log:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                qa_log.write(f"{user_id},{timestamp},{sample_idx+1},{rms},{centroid},{snr},{liveness_score}\n")
            if rms < 0.0007 or snr < 10:
                print(f"[Warning] Poor echo quality (RMS={rms:.6f}, SNR={snr:.2f} dB). Please reseat the ear-bud and stay still. Retrying sample {sample_idx+1}...")
                attempts += 1
                continue
            if liveness_score < -1:
                print("⚠️ Weak liveness score — possible spoof or echo fail.")
            # Save recording
            base_filename = f"user_{user_id}_echo_{timestamp}_{sample_idx+1:02d}"
            wav_path = os.path.join(echo_dir, base_filename + ".wav")
            meta_path = os.path.join(echo_dir, base_filename + "_meta.json")
            metadata = {
                "user_id": user_id,
                "timestamp": timestamp,
                "sample_number": sample_idx + 1,
                "tone_duration": tone_duration,
                "record_duration": record_duration,
                "freq": freq,
                "in_ear": True
            }
            save_recording_to_path(echo, wav_path, meta_path, metadata=metadata)
            print(f"[QA] Sample {sample_idx+1}: RMS={rms:.6f}, Centroid={centroid:.2f}, SNR={snr:.2f} dB, Liveness={liveness_score:.2f}")
            break  # Success, move to next sample
        else:
            print(f"[Error] Failed to collect valid sample {sample_idx+1} after 3 attempts. Skipping.")
        sample_idx += 1
        if sample_idx < num_samples:
            print("Waiting 2 seconds before next sample...")
            time.sleep(2)
    
    print("\nIn-ear sample collection complete.")

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