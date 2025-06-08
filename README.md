# Ear Canal + Voice Biometric Authentication System (Multi-Modal)

**This project uses both ear canal echo and voice features for robust, multi-modal biometric authentication. It is NOT ear-only: both modalities are required for best performance.**

# Ear Canal + Voice Acoustic Biometrics

This project implements a multi-modal acoustic-based biometric authentication system using both ear canal resonance patterns (echo) and voice/phrase audio. The system uses audio signals to capture unique acoustic signatures from a person's ear canal and their voice for secure authentication and liveness detection.

## Recent Changes (2025-06)
- System is now fully multi-modal: both ear canal echo and voice/phrase are used for authentication and liveness.
- Data collection scripts (`data_collector.py`, `phrase_collector.py`) refactored for robust path handling, metadata, and multi-modal support.
- Batch-based liveness/occlusion (open-air) test implemented for echo data collection.
- Chirp tone, increased tone duration, and RMS thresholding added for improved echo capture.
- Feature extraction (`enhanced_features.py`) now pairs in-ear/open-air samples, computes liveness score, and includes it as a feature.
- Analysis script (`analyze_liveness_comparison.py`) compares in-ear vs. open-air samples, visualizes waveforms, automates outlier detection, and adds soft liveness classification (logistic regression).
- STFT and magnitude are now reused for all spectral features, significantly speeding up feature extraction.
- All visualizations and analysis results are saved to organized folders for review.
- Documentation, code, and CLI updated to reflect multi-modal (ear + voice) nature throughout.

## Project Structure

```
ear_biometrics_prototype/
├── recordings/           # Raw audio recordings (echo/ and voice/)
├── data_collector.py    # Script for collecting ear canal echo samples
├── phrase_collector.py  # Script for collecting voice/phrase samples
├── enhanced_features.py # Multi-modal feature extraction
├── enhanced_train_model.py # Multi-modal model training
├── enhanced_predict.py  # Multi-modal prediction
├── utils/              # Utility functions for audio processing
│   ├── __init__.py
│   ├── audio.py       # Audio processing functions
│   └── features.py    # Feature extraction functions
├── performance_analysis/ # Metrics, confusion matrices, plots
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

To collect ear canal echo samples:

1. Connect your wired earphones/headset
2. Run the data collection script:
```bash
python data_collector.py --user_id USER_ID --samples 20
```

To collect voice/phrase samples:
```bash
python phrase_collector.py --user_id USER_ID --samples 3
```

This will:
- Play a test tone or prompt for phrase
- Record the echo response or voice
- Save the recordings in the `recordings/echo/` or `recordings/voice/` directory, with metadata

## Hardware Requirements

For best results, use:
- Wired earphones/headset with microphone
- USB audio interface (optional, for higher quality)

## Data Format

Each recording is saved as:
- WAV file (44.1kHz, 16-bit)
- Metadata JSON with user ID, timestamp, recording parameters, and phrase (for voice)

## Feature Extraction & Model Training

- Multi-modal feature extraction: Both echo and voice features are extracted, with `echo_` and `voice_` prefixes
- Liveness detection: Open-air (occlusion) samples are paired and scored
- Model training: Supports echo-only, voice-only, fused, and late/hybrid fusion modes
- Soft/automated liveness classification and outlier detection included

## Next Steps

- [x] Implement robust multi-modal feature extraction
- [x] Train ML models for echo, voice, and fused modalities
- [x] Add liveness/occlusion detection and soft classification
- [x] Optimize feature extraction speed (STFT reuse)
- [ ] Build authentication pipeline and mobile interface (future)

## Project Roadmap

- See `PROJECT_DOCUMENTATION.md` for full details, technical implementation, and future plans.

## Note

**This project is NOT ear-only. Both ear canal echo and voice/phrase modalities are required for best performance and security.**