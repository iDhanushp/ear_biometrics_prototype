import os
import numpy as np
import librosa

# Set up directories
ECHO_DIR = os.path.join('recordings', 'echo')
VOICE_DIR = os.path.join('recordings', 'voice')
SPEC_ECHO_DIR = os.path.join('spectrograms', 'echo')
SPEC_VOICE_DIR = os.path.join('spectrograms', 'voice')
os.makedirs(SPEC_ECHO_DIR, exist_ok=True)
os.makedirs(SPEC_VOICE_DIR, exist_ok=True)

# Parameters for mel-spectrogram
SR = 16000
N_MELS = 64
HOP_LENGTH = 256
N_FFT = 1024

# Helper to process a folder
def process_folder(src_dir, dst_dir):
    for fname in os.listdir(src_dir):
        if not fname.endswith('.wav'):
            continue
        src_path = os.path.join(src_dir, fname)
        y, sr = librosa.load(src_path, sr=SR)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
        S_db = librosa.power_to_db(S, ref=np.max)
        out_path = os.path.join(dst_dir, fname.replace('.wav', '.npy'))
        np.save(out_path, S_db)
        print(f"Saved: {out_path}")

if __name__ == '__main__':
    print('Extracting mel-spectrograms for echo...')
    process_folder(ECHO_DIR, SPEC_ECHO_DIR)
    print('Extracting mel-spectrograms for voice...')
    process_folder(VOICE_DIR, SPEC_VOICE_DIR)
    print('Done!')
