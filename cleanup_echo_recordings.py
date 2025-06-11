#!/usr/bin/env python3
"""
Echo Recording Quality Cleaner
------------------------------
Scans recordings/echo for in-ear samples that fail simple quality checks
(RMS < 4e-4 or |centroid-700| > 500 Hz or SNR < 10 dB) and moves both the
WAV and its companion _meta.json into recordings/echo/_rejected for manual
review. Use before training to prevent poor-quality echoes from hurting the
model.

Usage
-----
python cleanup_echo_recordings.py [--echo_dir recordings/echo]
"""
import argparse
import os
import shutil
from pathlib import Path
import librosa
import numpy as np
from scipy import signal

ECHO_RMS_MIN = 4e-4
CENTROID_TARGET = 700
CENTROID_TOL = 500
SNR_MIN_DB = 10


def compute_metrics(audio, sr=44100):
    """Return (rms, centroid, snr_db)."""
    rms = float(np.mean(librosa.feature.rms(y=audio)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    noise_win = audio[: int(0.05 * sr)]
    sig_win = audio[int(0.05 * sr) :]
    snr_db = 10 * np.log10(np.var(sig_win) / (np.var(noise_win) + 1e-9))
    return rms, centroid, snr_db


def is_bad(rms, centroid, snr_db):
    bad_rms = rms < ECHO_RMS_MIN
    bad_centroid = abs(centroid - CENTROID_TARGET) > CENTROID_TOL
    bad_snr = snr_db < SNR_MIN_DB
    return bad_rms or bad_centroid or bad_snr


def main(echo_dir: str):
    echo_path = Path(echo_dir)
    rejected_dir = echo_path / "_rejected"
    rejected_dir.mkdir(exist_ok=True)

    wav_files = list(echo_path.glob("*.wav"))
    total = len(wav_files)
    bad_count = 0

    for wav in wav_files:
        if "_openair" in wav.name:
            continue  # skip any open-air samples
        audio, sr = librosa.load(wav, sr=44100)
        rms, centroid, snr_db = compute_metrics(audio, sr)
        if is_bad(rms, centroid, snr_db):
            bad_count += 1
            # Move wav and meta file
            meta = wav.with_suffix("_meta.json")
            target_wav = rejected_dir / wav.name
            target_meta = rejected_dir / meta.name
            shutil.move(wav, target_wav)
            if meta.exists():
                shutil.move(meta, target_meta)
            print(f"Moved {wav.name} -> _rejected (RMS={rms:.6f}, Centroid={centroid:.2f}, SNR={snr_db:.2f} dB)")

    print(f"\nScan complete. {bad_count}/{total} files moved to {rejected_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move low-quality echo recordings to _rejected folder.")
    parser.add_argument("--echo_dir", default="recordings/echo", help="Directory containing echo WAV files")
    args = parser.parse_args()
    main(args.echo_dir) 