# Ear Canal + Voice Biometric Authentication System (Multi-Modal)

**This project uses both ear canal echo and voice features for robust, multi-modal biometric authentication. It is NOT ear-only: both modalities are required for best performance.**

# Ear Canal Acoustic Biometrics

This project implements an acoustic-based biometric authentication system using ear canal resonance patterns. The system uses audio signals to capture unique acoustic signatures from a person's ear canal for secure authentication.

## Project Structure

```
ear_biometrics_prototype/
├── recordings/           # Raw audio recordings
├── data_collector.py    # Script for collecting ear canal audio samples
├── utils/              # Utility functions for audio processing
│   ├── __init__.py
│   ├── audio.py       # Audio processing functions
│   └── features.py    # Feature extraction functions
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Collection

To collect ear canal audio samples:

1. Connect your wired earphones/headset
2. Run the data collection script:
```bash
python data_collector.py --user_id USER_ID --samples 20
```

This will:
- Play a test tone through the earphones
- Record the echo response
- Save the recordings in the `recordings/` directory

## Hardware Requirements

For best results, use:
- Wired earphones/headset with microphone
- USB audio interface (optional, for higher quality)

## Data Format

Each recording is saved as:
- WAV file (44.1kHz, 16-bit)
- Metadata JSON with:
  - User ID
  - Timestamp
  - Recording parameters

## Next Steps

- [ ] Implement feature extraction
- [ ] Train ML model
- [ ] Build authentication pipeline
- [ ] Create Flutter app interface 

## Project Roadmap

Below is a step-by-step plan to build an acoustic-based biometric authentication system (using ear canal resonance and voice) from prototype to testing:

### STAGE 1: Plan the Prototype

- **Hardware (Choose One for Prototyping):**
  - Wired in-ear headset (3.5mm or USB-C) – low cost, raw access.
  - USB headset (with mic) – great for laptop testing.
  - Optional: Audio interface (e.g., Focusrite) – for high-precision audio.

### STAGE 2: Build the Data Collection System

- Use Python (with libraries such as pyaudio or sounddevice) to record echo responses from the ear.
- Play a short tone (e.g., chirp or sine burst) and record the reflected sound via the mic.
- Save recordings (WAV) and metadata (JSON) in a "recordings/" folder (e.g., user_id, timestamp, mic_response.wav).
- Collect 20–30 samples from 10–15 users.

### STAGE 3: Feature Extraction

- Use signal processing (FFT, MFCC, spectral features) to convert each recorded response into ML features.
- Libraries: librosa, scipy.signal, numpy.

### STAGE 4: Train a Classifier

- Train a model (e.g., KNN, SVM, Random Forest, or CNN) to recognize users based on ear canal echo.
- Use scikit-learn (for classical models) or TensorFlow/PyTorch (for deep learning).

### STAGE 5: Testing and Evaluation

- Evaluate using metrics (Accuracy, FAR, FRR) and robustness tests (e.g., slight earbud shift, added noise).
- Use sklearn.metrics (classification_report, confusion_matrix) for evaluation.

### STAGE 6: Expand and Polish

- Build a Flutter front-end (or integrate with a mobile app) to simulate a login screen.
- Experiment with different impulse sounds (chirps, noise bursts) and consider multi-modal fusion (voice + ear echo).
- Research privacy and spoofing resistance.

### DATASET

- No public ear-canal acoustic dataset exists; you will need to collect your own data (which is ideal for your hardware and use case).

### TOOLS RECAP

- **Python:** Recording, processing, and ML.
- **librosa:** Audio feature extraction.
- **scikit-learn:** Training classical models.
- **PyTorch/TensorFlow:** Deep learning (optional).
- **Flutter:** Mobile app integration.