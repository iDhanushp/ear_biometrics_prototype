#!/usr/bin/env python3
import sounddevice as sounddev
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
from utils.audio import select_audio_device, select_audio_input_device, robust_play, robust_record_echo, interactive_device_test, generate_test_beep, test_all_input_devices, generate_cue_beep
import os
from datetime import datetime
import json
from scipy import signal
import librosa
from scipy.signal import correlate

def generate_chirp_tone(duration=0.5, sr=44100, f_start=300, f_end=4000, amplitude=0.8):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    chirp = signal.chirp(t, f0=f_start, f1=f_end, t1=duration, method='linear')
    chirp = chirp * amplitude
    return chirp.astype(np.float32)

def align_echo(tone, recorded):
    corr = correlate(recorded, tone, mode='full')
    lag = np.argmax(corr) - len(tone)
    aligned_tail = recorded[lag + len(tone):]
    return aligned_tail, lag

def collect_samples(user_id: str, num_samples: int = 20, tone_duration: float = 0.5,
                   record_duration: float = 0.7, freq: int = 1000, output_device=None, input_device=None,
                   playback_samplerate=44100, record_samplerate=44100):
    """Collect ear canal echo samples for a user.
    
    Args:
        user_id (str): User identifier
        num_samples (int): Number of samples to collect
        tone_duration (float): Duration of test tone in seconds
        record_duration (float): Duration of echo recording in seconds
        freq (int): Frequency of test tone in Hz
        output_device (int): Output device index
        input_device (int): Input device index
        playback_samplerate (int): Sample rate for playback
        record_samplerate (int): Sample rate for recording
    """
    print(f"\nCollecting {num_samples} in-ear samples for user {user_id}")
    print("Please wear your earphones/headset and stay in a quiet environment.")
    print("Press Enter when ready...")
    input()
    # Use chirp tone for broader resonance
    tone = generate_chirp_tone(duration=tone_duration, sr=playback_samplerate)
    
    # Ensure echo subfolder exists
    echo_dir = os.path.join('recordings', 'echo')
    os.makedirs(echo_dir, exist_ok=True)
    
    # Confirm audio output with a test beep
    print("\nPlaying a test beep to confirm audio output. Please listen...")
    cue_beep = generate_test_beep(duration=1.0, sr=playback_samplerate, amplitude=1.0)
    print("[INFO] If you do not hear the beep, check that your headset is in 'Stereo' (A2DP) mode, not 'Hands-Free' (HFP/HSP). Also check Windows volume.")
    # Interactive device test if no output or user requests
    if output_device is None:
        print("\nYou will now cycle through all available output devices.\n")
        output_device = interactive_device_test(cue_beep, samplerate=playback_samplerate)
        if output_device is None:
            print("No working output device found. Aborting.")
            return
    else:
        try:
            if output_device is not None:
                print(f"[DEBUG] Playing test beep on output device index: {output_device}")
            else:
                print("[DEBUG] Playing test beep on system default output device")
            robust_play(cue_beep, samplerate=playback_samplerate, device=output_device)
        except Exception as e:
            print(f"[ERROR] Could not play test beep: {e}")
            print("Aborting due to repeated playback failure.")
            return
    print("If you did not hear the beep, check your audio device and press Enter to try again, or Ctrl+C to abort.")
    input()
    
    # Open QA log CSV
    qa_log_path = os.path.join(echo_dir, f"qa_log_{user_id}.csv")
    with open(qa_log_path, "a") as qa_log:
        qa_log.write("user_id,timestamp,sample_number,rms,zcr,decay,lag_offset_ms,echo_tail_ms\n")
    
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
            # Play a cue beep (distinct, 330 Hz, 0.7s, stereo, full amplitude)
            cue_beep = generate_cue_beep(duration=0.7, sr=playback_samplerate, amplitude=1.0)
            try:
                print(f"[DEBUG] {time.strftime('%H:%M:%S')} Playing cue beep on output device index: {output_device}")
                robust_play(cue_beep, samplerate=playback_samplerate, device=output_device)
            except Exception as e:
                print(f"[ERROR] Could not play cue beep: {e}")
                print("Aborting due to playback failure.")
                return
            time.sleep(0.5)  # Ensure cue beep finishes
            # Use a longer, louder, lower-frequency chirp for robust in-ear pickup
            chirp_duration = 2.0
            chirp_amplitude = 1.0  # maximize amplitude, clip below
            chirp_f_start = 500
            chirp_f_end = 900
            tone_play = generate_chirp_tone(duration=chirp_duration, sr=playback_samplerate, f_start=chirp_f_start, f_end=chirp_f_end, amplitude=chirp_amplitude)
            tone_play = np.clip(tone_play, -0.99, 0.99)
            # Ensure chirp is stereo for playback (matches cue beep)
            if tone_play.ndim == 1:
                tone_play = np.column_stack([tone_play, tone_play])
            print(f"[DEBUG] Chirp/test tone shape for playback: {tone_play.shape}, min={np.min(tone_play):.4f}, max={np.max(tone_play):.4f}")
            # --- Pre-tone delay: record first, then play tone, then continue recording ---
            print(f"[DEBUG] {time.strftime('%H:%M:%S')} Pre-tone delay: using full-duplex Stream with callback for echo capture.")
            try:
                pre_tone_delay = 0.0  # No pre-tone delay
                post_tone_delay = 3.0  # Increase to 3 seconds for robust echo tail capture
                total_record_duration = pre_tone_delay + chirp_duration + post_tone_delay
                silence_pre = np.zeros((int(pre_tone_delay * playback_samplerate), 2), dtype=np.float32)
                silence_post = np.zeros((int(post_tone_delay * playback_samplerate), 2), dtype=np.float32)
                # Ensure tone_play is stereo
                if tone_play.ndim == 1:
                    tone_play = np.column_stack([tone_play, tone_play])
                play_buffer = np.concatenate([silence_pre, tone_play, silence_post], axis=0)
                input_buffer = np.zeros(play_buffer.shape[0], dtype=np.float32)
                def audio_callback(indata, outdata, frames, time, status):
                    start = audio_callback.frame
                    end = start + frames
                    outdata[:] = play_buffer[start:end]
                    input_buffer[start:end] = indata[:, 0]  # Use first channel
                    audio_callback.frame += frames
                audio_callback.frame = 0
                with sounddev.Stream(device=(input_device, output_device),
                                    samplerate=playback_samplerate,
                                    channels=(1, 2),
                                    dtype='float32',
                                    callback=audio_callback,
                                    blocksize=1024):
                    sounddev.sleep(int(1000 * play_buffer.shape[0] / playback_samplerate))
                echo = input_buffer
                print(f"[DEBUG] Raw echo (stream callback): min={{np.min(echo):.6f}}, max={{np.max(echo):.6f}}, mean={{np.mean(echo):.6f}}")
            except Exception as e:
                print(f"[ERROR] Stream callback failed: {e}")
                return
            # Validate raw echo before any further processing or saving
            if not np.all(np.isfinite(echo)):
                print(f"[ERROR] Raw echo contains NaN or Inf values. Skipping this sample.")
                continue  # Skip to next attempt/sample
            # Save raw echo recording for manual inspection (temporary)
            import scipy.io.wavfile
            raw_echo_path = os.path.join(echo_dir, f"raw_echo_sample_{sample_idx+1}.wav")
            scipy.io.wavfile.write(raw_echo_path, 44100, echo.astype(np.float32))
            print(f"[DEBUG] Saved raw echo to {raw_echo_path}")
            # Validate raw echo before any further processing or saving
            if not np.all(np.isfinite(echo)):
                print(f"[ERROR] Raw echo contains NaN or Inf values. Skipping this sample.")
                continue  # Skip to next attempt/sample
            # Resample echo to 16kHz for all processing if needed
            if 44100 != 16000:
                echo_16k = librosa.resample(echo, orig_sr=44100, target_sr=16000)
            else:
                echo_16k = echo
            
            # Band-pass filter (300–4000 Hz) for echo tail
            from scipy.signal import butter, filtfilt
            def bandpass(data, sr, low=300, high=4000, order=4):
                nyq = 0.5 * sr
                b, a = butter(order, [low/nyq, high/nyq], btype='band')
                return filtfilt(b, a, data)
            echo_16k = bandpass(echo_16k, 16000)
            # Smooth echo envelope (Savitzky–Golay)
            from scipy.signal import savgol_filter
            envelope = np.abs(echo_16k)
            envelope = savgol_filter(envelope, 101, 3)  # window size 101, polyorder 3
            # Compute RMS, ZCR, decay, lag on 16kHz, bandpassed, smoothed echo
            rms = float(np.mean(envelope))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(echo_16k, frame_length=400, hop_length=160)))
            # Echo tail energy decay (ratio of last 25% to first 25%)
            n = len(envelope)
            decay = float(np.sum(envelope[int(0.75*n):]) / (np.sum(envelope[:int(0.25*n)]) + 1e-9))
            # Normalized cross-correlation lag (in ms)
            lag = np.argmax(np.correlate(echo_16k, echo_16k, mode='full')) - len(echo_16k)
            lag_offset_ms = 1000 * lag / 16000
            echo_tail_ms = 1000 * len(echo_16k) / 16000
            # Save failed recordings for analysis
            if echo_tail_ms < 150:
                import scipy.io.wavfile
                scipy.io.wavfile.write(f"failed_sample_{sample_idx+1}.wav", 44100, echo.astype(np.float32))
            # Save QA log
            with open(qa_log_path, "a") as qa_log:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                qa_log.write(f"{user_id},{timestamp},{sample_idx+1},{rms},{zcr},{decay},{lag_offset_ms},{echo_tail_ms}\n")
            # Quality checks (example: adjust as needed)
            if echo_tail_ms < 150:
                print(f"[Warning] Echo tail < 150ms after alignment. Retrying sample {sample_idx+1}...")
                attempts += 1
                continue
            if rms < 0.0007:
                print(f"[Warning] Low RMS. Retrying sample {sample_idx+1}...")
                attempts += 1
                continue
            # Save recording (aligned echo only, 16kHz)
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
                "in_ear": True,
                "lag_offset_ms": lag_offset_ms,
                "echo_tail_ms": echo_tail_ms,
                "zcr": zcr,
                "decay": decay,
                "rms": rms
            }
            save_recording_to_path(echo_16k, wav_path, meta_path, metadata=metadata)
            print(f"[QA] Sample {sample_idx+1}: RMS={rms:.6f}, ZCR={zcr:.4f}, Decay={decay:.4f}, Lag={lag_offset_ms:.1f} ms, Tail={echo_tail_ms:.1f} ms")
            # Save raw echo recording for manual inspection (temporary)
            import scipy.io.wavfile
            raw_echo_path = os.path.join(echo_dir, f"raw_echo_sample_{sample_idx+1}.wav")
            scipy.io.wavfile.write(raw_echo_path, 44100, echo.astype(np.float32))
            print(f"[DEBUG] Saved raw echo to {raw_echo_path}")
            break  # Success, move to next sample
        else:
            print(f"[Error] Failed to collect valid sample {sample_idx+1} after 3 attempts. Skipping.")
        sample_idx += 1
        if sample_idx < num_samples:
            print("Waiting 2 seconds before next sample...")
            time.sleep(2)
    
    print("\nIn-ear sample collection complete.")

def main():
    print("[DEBUG] main() started")
    parser = argparse.ArgumentParser(description="Collect ear canal echo samples for biometric authentication")
    parser.add_argument("--user_id", help="User identifier")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to collect")
    parser.add_argument("--tone_duration", type=float, default=0.5, help="Duration of test tone in seconds")
    parser.add_argument("--record_duration", type=float, default=0.7, help="Duration of echo recording in seconds")
    parser.add_argument("--freq", type=int, default=1000, help="Frequency of test tone in Hz")
    parser.add_argument("--list_devices", action="store_true", help="List available audio devices")
    parser.add_argument("--output_device", type=int, default=None, help="Output device index for playback/recording")
    parser.add_argument("--input_device", type=int, default=None, help="Input device index for recording")
    parser.add_argument("--test_inputs", action="store_true", help="Test all input devices before selecting")
    parser.add_argument("--open_air", action="store_true", help="Record open-air (bud out) echo for differential scoring")
    args = parser.parse_args()
    print("[DEBUG] Arguments parsed")

    if args.list_devices:
        print("[DEBUG] Listing devices and exiting")
        list_audio_devices()
        return

    if not args.user_id:
        print("[DEBUG] No user_id provided, error")
        parser.error("--user_id is required unless --list_devices is used")

    # --- DEVICE SELECTION LOGIC ---
    # Uncomment the block for the hardware you want to use

    # --- Laptop built-in mic/speakers (Windows WASAPI) ---
    # output_device = 11  # Speakers (2- Realtek(R) Audio), Windows WASAPI (0 in, 2 out)
    # input_device = 12   # Microphone Array (2- Intel® Smart Sound Technology for Digital Microphones), Windows WASAPI (2 in, 0 out)
    # out_info = sounddev.query_devices(output_device, 'output')
    # in_info = sounddev.query_devices(input_device, 'input')
    # playback_samplerate = int(out_info['default_samplerate'])
    # record_samplerate = int(in_info['default_samplerate'])
    # print(f"[INFO] Using Laptop: output_device={output_device} (Speakers), input_device={input_device} (Microphone Array)")
    # print(f"[INFO] Playback sample rate: {playback_samplerate} Hz | Record sample rate: {record_samplerate} Hz")

    # --- Muff M2 TWS (MME) ---
    # output_device = 4  # Headphones (Muffs M2), MME (0 in, 2 out)
    # input_device = 2   # Headset (Muffs M2), MME (1 in, 0 out)
    # print("[INFO] Using Muff M2: output_device=4 (Headphones), input_device=2 (Headset)")
    # print("[WARNING] On Windows, you cannot use stereo output and mic input simultaneously on most TWS. If you get low RMS or NaN, try using only one earbud, or set Boult as Default Communication Device in Windows Sound settings.")

    # --- Boult TWS (WASAPI) ---
    # import sounddevice as sd
    # sd.default.device = (18, 14)  # WASAPI: Headset (Boult Audio Airbass), Headphones (Boult Audio Airbass)
    # print("[INFO] sounddevice set to WASAPI Boult devices (input=18, output=14)")

    # --- Use Microsoft Sound Mapper - Output as default output device ---
    # output_device = 3  # Microsoft Sound Mapper - Output
    # print(f"[INFO] Using output_device=3 (Microsoft Sound Mapper - Output)")
    # Keep input_device as system default
    # input_device = None
    # Query default sample rates
    # in_info = sounddev.query_devices(None, 'input')
    # out_info = sounddev.query_devices(output_device, 'output')
    # playback_samplerate = int(out_info['default_samplerate'])
    # record_samplerate = int(in_info['default_samplerate'])
    # print(f"[INFO] Playback sample rate: {playback_samplerate} Hz | Record sample rate: {record_samplerate} Hz")

    # --- Use specific devices: Realtek mic (MME) for input, Muff/Boult for output ---
    input_device = 1  # Microphone (2- Realtek(R) Audio), MME (2 in)
    output_device = 5  # Headphones (Boult Audio Airbass), MME (2 out)
    print(f"[INFO] Using input_device=1 (Microphone (2- Realtek(R) Audio), MME), output_device=5 (Headphones (Boult Audio Airbass), MME)")
    # Query sample rates
    in_info = sounddev.query_devices(input_device, 'input')
    out_info = sounddev.query_devices(output_device, 'output')
    playback_samplerate = int(out_info['default_samplerate'])
    record_samplerate = int(in_info['default_samplerate'])
    print(f"[INFO] Playback sample rate: {playback_samplerate} Hz | Record sample rate: {record_samplerate} Hz")

    # Ensure no global device override is set
    sounddev.default.device = None  # Always use explicit device indices below
    print("[DEBUG] Cleared sd.default.device to avoid global override.")

    # Query and print supported sample rates for selected devices
    import sounddevice as sd
    out_info = sd.query_devices(output_device, 'output')
    in_info = sd.query_devices(input_device, 'input')
    print(f"[DEBUG] Output device {output_device}: {out_info['name']} | Supported sample rate: {out_info['default_samplerate']}")
    print(f"[DEBUG] Input device {input_device}: {in_info['name']} | Supported sample rate: {in_info['default_samplerate']}")
    
    collect_samples(
        user_id=args.user_id,
        num_samples=args.samples,
        tone_duration=args.tone_duration,
        record_duration=args.record_duration,
        freq=args.freq,
        output_device=output_device,
        input_device=input_device,
        playback_samplerate=playback_samplerate,
        record_samplerate=record_samplerate
    )
    print("[DEBUG] collect_samples() returned")

if __name__ == "__main__":
    main()