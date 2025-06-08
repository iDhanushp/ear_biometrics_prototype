import os
import json
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_metadata(meta_path):
    with open(meta_path, 'r') as f:
        return json.load(f)

def get_sample_pairs(echo_dir):
    """
    Find and pair in-ear and open-air samples by user, timestamp, and sample_number.
    Returns a list of tuples: (in_ear_wav, open_air_wav, in_ear_meta, open_air_meta)
    """
    pairs = []
    for fname in os.listdir(echo_dir):
        if fname.endswith('.wav') and '_openair' not in fname:
            base = fname[:-4]
            openair_fname = base + '_openair.wav'
            inear_meta = os.path.join(echo_dir, base + '_meta.json')
            openair_meta = os.path.join(echo_dir, base + '_openair_meta.json')
            openair_path = os.path.join(echo_dir, openair_fname)
            inear_path = os.path.join(echo_dir, fname)
            if os.path.exists(openair_path) and os.path.exists(inear_meta) and os.path.exists(openair_meta):
                pairs.append((inear_path, openair_path, inear_meta, openair_meta))
    return pairs

def extract_rms(audio, sr=44100):
    return float(np.mean(librosa.feature.rms(y=audio)))

def plot_waveform_and_spectrogram(audio, sr, title, save_dir=None):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(audio)) / sr, audio)
    plt.title(f"Waveform: {title}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    # Spectrogram
    plt.subplot(1, 2, 2)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log')
    plt.title(f"Spectrogram: {title}")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = title.replace(' ', '_').replace(':', '').replace('/', '_')
        plt.savefig(os.path.join(save_dir, f"{fname}_waveform_spectrogram.png"))
    plt.close()

def plot_fft(audio, sr, title, save_dir=None):
    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1/sr)
    plt.figure(figsize=(8, 3))
    plt.plot(freqs, fft)
    plt.title(f"FFT Spectrum: {title}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = title.replace(' ', '_').replace(':', '').replace('/', '_')
        plt.savefig(os.path.join(save_dir, f"{fname}_fft.png"))
    plt.close()

def extract_spectral_centroid(audio, sr=44100):
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    return float(np.mean(centroid))

def extract_resonance_peaks(audio, sr=44100, num_peaks=3):
    freqs, fft_magnitude = np.fft.rfftfreq(len(audio), 1/sr), np.abs(np.fft.rfft(audio))
    peaks = np.argpartition(fft_magnitude, -num_peaks)[-num_peaks:]
    sorted_peaks = peaks[np.argsort(fft_magnitude[peaks])[::-1]]
    return freqs[sorted_peaks], fft_magnitude[sorted_peaks]

def is_live_soft(rms_diff, centroid_diff):
    # More forgiving logic
    if rms_diff > 0.0001 and abs(centroid_diff) < 300:
        return True
    elif rms_diff > -0.0002 and abs(centroid_diff) < 600:
        return "unsure"  # Ask for retry
    else:
        return False

def main():
    echo_dir = os.path.join('recordings', 'echo')
    plot_dir = os.path.join('performance_analysis', 'visualizations', 'liveness_waveforms')
    pairs = get_sample_pairs(echo_dir)
    print(f"Found {len(pairs)} in-ear/open-air sample pairs.")
    results = []
    for idx, (inear_wav, openair_wav, inear_meta, openair_meta) in enumerate(pairs):
        inear_audio, sr = librosa.load(inear_wav, sr=44100)
        openair_audio, _ = librosa.load(openair_wav, sr=44100)
        inear_rms = extract_rms(inear_audio, sr)
        openair_rms = extract_rms(openair_audio, sr)
        rms_diff = inear_rms - openair_rms
        liveness_score = (inear_rms - openair_rms) / (inear_rms + 1e-6)
        inear_info = load_metadata(inear_meta)
        # Spectral centroid
        inear_centroid = extract_spectral_centroid(inear_audio, sr)
        openair_centroid = extract_spectral_centroid(openair_audio, sr)
        centroid_diff = inear_centroid - openair_centroid
        # Resonance peaks
        inear_peaks_freq, inear_peaks_mag = extract_resonance_peaks(inear_audio, sr)
        openair_peaks_freq, openair_peaks_mag = extract_resonance_peaks(openair_audio, sr)
        results.append({
            'user_id': inear_info['user_id'],
            'timestamp': inear_info['timestamp'],
            'sample_number': inear_info['sample_number'],
            'rms_in_ear': inear_rms,
            'rms_open_air': openair_rms,
            'rms_diff': rms_diff,
            'liveness_score': liveness_score,
            'centroid_in_ear': inear_centroid,
            'centroid_open_air': openair_centroid,
            'centroid_diff': centroid_diff,
            'peaks_in_ear_freq': inear_peaks_freq.tolist(),
            'peaks_in_ear_mag': inear_peaks_mag.tolist(),
            'peaks_open_air_freq': openair_peaks_freq.tolist(),
            'peaks_open_air_mag': openair_peaks_mag.tolist()
        })
        # Save plots for the first pair (or all pairs if you want)
        if idx == 0:
            plot_waveform_and_spectrogram(inear_audio, sr, f"In-Ear_Sample_{inear_info['sample_number']}", save_dir=plot_dir)
            plot_waveform_and_spectrogram(openair_audio, sr, f"Open-Air_Sample_{inear_info['sample_number']}", save_dir=plot_dir)
            plot_fft(inear_audio, sr, f"In-Ear_Sample_{inear_info['sample_number']}", save_dir=plot_dir)
            plot_fft(openair_audio, sr, f"Open-Air_Sample_{inear_info['sample_number']}", save_dir=plot_dir)
    df = pd.DataFrame(results)
    print(df[['user_id', 'sample_number', 'rms_in_ear', 'rms_open_air', 'rms_diff', 'liveness_score', 'centroid_in_ear', 'centroid_open_air', 'centroid_diff']])

    # Resonance peak analysis: print and plot peak frequencies and magnitudes
    print("\nResonance Peaks (Top 3) for each sample:")
    for i, row in df.iterrows():
        print(f"Sample {row['sample_number']}: In-ear peaks {row['peaks_in_ear_freq']} (mag {row['peaks_in_ear_mag']}), Open-air peaks {row['peaks_open_air_freq']} (mag {row['peaks_open_air_mag']})")
    # Plot resonance peak frequencies for in-ear vs open-air
    plt.figure(figsize=(10, 5))
    for i, row in df.iterrows():
        plt.scatter([1,2,3], row['peaks_in_ear_freq'], color='b', alpha=0.6, label='In-Ear' if i==0 else "")
        plt.scatter([1,2,3], row['peaks_open_air_freq'], color='r', alpha=0.6, label='Open-Air' if i==0 else "")
    plt.xticks([1,2,3], ['Peak 1', 'Peak 2', 'Peak 3'])
    plt.ylabel('Frequency (Hz)')
    plt.title('Top 3 Resonance Peak Frequencies: In-Ear vs Open-Air')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Outlier detection: flag samples with negative liveness_score or centroid_diff
    df['outlier'] = (df['liveness_score'] < 0) | (df['centroid_diff'] < 0)
    print("\nOutlier samples (possible bad fit, noise, or failed echo):")
    print(df[df['outlier']][['user_id', 'sample_number', 'liveness_score', 'centroid_diff']])
    # Save to CSV for further analysis
    df.to_csv('liveness_comparison_results.csv', index=False)
    print('Results saved to liveness_comparison_results.csv')

    # Soft classification for liveness (logistic regression)
    # Label: 1 if liveness_score > 0 and centroid_diff > -100, else 0 (for demo)
    df['label'] = ((df['liveness_score'] > 0) & (df['centroid_diff'] > -100)).astype(int)
    X = df[['rms_diff', 'centroid_diff']].values
    y = df['label'].values
    if len(df) > 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("\nSoft Liveness Classifier Report:")
        print(classification_report(y_test, y_pred))
        df['soft_liveness_prob'] = clf.predict_proba(X)[:,1]
    else:
        print("Not enough samples for soft classifier training.")

    # Log all low-confidence real samples
    low_conf = df[(df['soft_liveness_prob'] < 0.6) & (df['label'] == 1)]
    if not low_conf.empty:
        print("\nLow-confidence real (possible false rejection) samples:")
        print(low_conf[['user_id', 'sample_number', 'rms_diff', 'centroid_diff', 'soft_liveness_prob']])
        low_conf.to_csv('false_rejection_set.csv', index=False)
        print('Low-confidence real samples saved to false_rejection_set.csv')

if __name__ == '__main__':
    main()
