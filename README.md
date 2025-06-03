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