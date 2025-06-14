# Ear Canal + Voice Biometric Authentication System (Multi-Modal)

**This project implements robust, multi-modal biometric authentication using both ear canal echo and voice features. It is NOT ear-only: both modalities are required for best performance.**

# Project Documentation

## Recent Changes (2025-06)
- System is now fully multi-modal: both ear canal echo and voice/phrase are used for authentication and liveness.
- Data collection scripts (`data_collector.py`, `phrase_collector.py`) refactored for robust path handling, metadata, and multi-modal support.
- Batch-based liveness/occlusion (open-air) test implemented for echo data collection. **[REMOVED: Open-air sample collection is no longer required; only in-ear samples are now collected for echo. Liveness detection is still supported via feature engineering and analysis.]**
- Chirp tone, increased tone duration, and RMS thresholding added for improved echo capture.
- Feature extraction (`enhanced_features.py`) now pairs in-ear/open-air samples, computes liveness score, and includes it as a feature.
- Analysis script (`analyze_liveness_comparison.py`) compares in-ear vs. open-air samples, visualizes waveforms, automates outlier detection, and adds soft liveness classification (logistic regression).
- STFT and magnitude are now reused for all spectral features, significantly speeding up feature extraction.
- **Adaptive noise reduction (spectral gating) is now integrated as a pre-processing step in feature extraction, improving robustness to background noise. Advanced ML-based denoising (Demucs) was tested but removed due to negligible benefit and high computational cost.**
- All visualizations and analysis results are saved to organized folders for review.
- Documentation, code, and CLI updated to reflect multi-modal (ear + voice) nature throughout.
- **Demucs and its dependencies have been removed from requirements.txt and the codebase. Only classical denoising is now supported.**
- Data collection no longer prompts for open-air (liveness) samples; only in-ear samples are collected for echo. Liveness detection is handled in feature extraction and analysis.
- Echo Signal Simulation (Data Quality): Done. Echo signal quality simulation/checks are implemented in `data_collector.py` and are part of the QA logging and fallback pipeline. This ensures robust data quality and device-agnostic operation.
- **Enhanced Echo Capture & Quality Control**: `data_collector.py` now uses an amplified chirp tone (amplitude 0.8) for better ear canal excitation. It also enforces stricter quality checks during recording, including a higher RMS threshold (0.0007) and a new Signal-to-Noise Ratio (SNR) threshold (10 dB). If quality is low, the script prompts the user to reseat the ear-bud and retries. RMS, centroid, and SNR are now logged in the QA CSV for detailed review.
- **Automated Echo Recording Cleanup**: A new script `cleanup_echo_recordings.py` has been added. This utility scans the `recordings/echo` directory and automatically moves any low-quality echo recordings (based on RMS, spectral centroid, and SNR thresholds) to a `_rejected` subfolder. This helps ensure that only high-quality echo data is used for model training. The script also handles creation of necessary parent directories.
- **Feature Extraction Quality Filtering**: `enhanced_features.py` now includes a filtering step during dataset creation. Echo recordings are automatically skipped if their `echo_rms_mean` is below `4e-4` or if their `echo_spectral_centroid_mean` deviates by more than `500 Hz` from the target `700 Hz`. This ensures that only good quality echo features contribute to the model training.

## Recent Codebase Changes (June 2024)

- **Command-line Arguments**: `enhanced_train_model.py` now supports `--feature_mode` (fused, echo, voice) and `--recordings_dir` for flexible training.
- **Voice Feature Extraction**: The feature extraction pipeline now always extracts both echo and voice features from all recordings, regardless of metadata. This ensures voice features are available for training and evaluation.
- **Expanded Voice Features**: Voice features now include MFCCs, LPC, chroma, pitch, formants, shimmer, jitter, and HNR, in addition to previously used features.
- **User Filtering**: During data preparation, users with fewer than 2 samples are automatically removed to prevent errors during stratified train/test splitting.
- **Classification Report Fix**: The results analysis now only reports on classes present in the test set, preventing errors if some users are filtered out.
- **MLP Hyper-parameter Grid**: The neural network (MLP) model's `batch_size` hyper-parameter grid is now limited to ['auto', 8, 16, 32] to avoid warnings about batch size exceeding the number of samples in a fold.
- **General Improvements**: Improved warning suppression, reporting, and code robustness for small or imbalanced datasets.

## June 2025: Major Feature and Tuning Upgrades
    - Added spectral entropy and spectral flux to feature extraction for improved voice discrimination.
    - Feature selection now supports RFECV (recursive feature elimination with cross-validation) in addition to RFE and SelectKBest. CLI exposes --feature_selection and --num_features.
    - SVM grid expanded to include poly/sigmoid kernels and more gamma/degree values. Neural network grid now supports deeper architectures.
    - All new options are documented in CLI and usage instructions below.

## June 2025: Major Pipeline, Device, and Robustness Upgrades

### Echo Data Collection Pipeline: Implementation, Debugging, and Robust Tuning

#### Key Changes & Solutions
- **Full EchoID TWS Compliance:**
  - Strict echo/voice separation, echo alignment, QA logging, and documentation updates.
  - Chirp redesigned to 500–900 Hz, amplitude maximized, duration extended, 16kHz/44.1kHz/48kHz pipeline enforced, bandpass and smoothing added.
  - Feature extraction switched to time-domain features (RMS, ZCR, decay, lag); QA log and checks updated.
- **Device Selection & Handling:**
  - Interactive and automatic device selection, fallback to system default, and robust error/debug output.
  - Robust playback/recording logic with fallback, stereo enforcement, and device/channel info printing.
  - Added code to always use output device 4 (Boult/Muff headphones), input device 2 (Boult/Muff mic), or other indices as needed, with fallback to other mics if needed.
  - Added code to print and validate input device sample rate and format, and to force mono or stereo input as needed.
  - Added code to clip audio before saving to WAV to prevent overflow warnings.
  - Added debug output for raw echo stats and error handling for NaN/Inf.
  - Pre-recording delay and longer recording duration added for Bluetooth latency; post-tone delay increased for robust echo tail capture.
  - Code now supports and prints input device sample rate and format, and can force 16kHz/44.1kHz/48kHz input for Boult mic.
  - Added "no playback" test mode and restored echo playback/recording logic with timing tweaks.
  - Tool calls: `python -m sounddevice` (device listing), robust debug/validation code.
- **Full-Duplex Streaming:**
  - Switched to `sounddevice.Stream` with callback for professional, tightly synchronized playback and recording (like DAWs/games/calls).
  - Uses stereo for playback and recording, processes only the first channel for echo.
  - Pre-tone delay removed for efficiency; post-tone delay set to 3s for robust echo tail capture.
- **Default Device Handling:**
  - Added option to use system default input/output devices, or explicitly select devices (e.g., Microsoft Sound Mapper, Boult, laptop mic/speakers).
  - Sample rates are always queried and matched to device capabilities.
- **QA & Logging:**
  - All raw echoes are saved for manual inspection.
  - RMS threshold can be tuned for debugging; QA log includes all relevant metrics.

#### Challenges Faced
- **Bluetooth TWS on Windows:**
  - Most Bluetooth TWS devices cannot do high-quality stereo playback and mic input simultaneously on Windows due to OS/driver limitations (Hands-Free Profile/HFP mode).
  - Boult/Muff TWS mics often capture only the played tone, with very weak or no echo tail.
  - Wired or built-in devices provide much better echo capture.
- **Device Index Shifts:**
  - Device indices change when devices are connected/disconnected; robust device listing and selection logic was required.
- **Sample Rate Mismatches:**
  - Many devices only support 48000 Hz or 44100 Hz; code now always queries and matches device sample rates.
- **NaN/Inf in Recordings:**
  - Caused by device/channel/sample rate mismatches or permissions; fixed by matching channels, checking permissions, and robust error handling.
- **Echo Tail Capture:**
  - If recording stops immediately after the tone, echo tail is lost; fixed by increasing post-tone delay.
- **Low RMS/Signal:**
  - Some devices (especially Bluetooth) produce low RMS; code now allows threshold tuning and always saves raw echoes for analysis.

#### Summary
- The system is now robust, EchoID TWS-compliant, and highly instrumented for debugging and QA, with special handling for Bluetooth quirks, Windows audio device issues, and fallback to laptop hardware.
- Full-duplex streaming and device/sample rate auto-detection ensure maximum compatibility and data quality.
- All changes, solutions, and challenges are documented for future reference and reproducibility.

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Data Collection Process](#data-collection-process)
4. [Feature Engineering](#feature-engineering)
5. [Multi-Modal Feature Extraction & Fusion](#multi-modal-feature-extraction--fusion)
6. [Model Development & Fusion Modes](#model-development--fusion-modes)
7. [Performance Analysis](#performance-analysis)
8. [Technical Implementation](#technical-implementation)
9. [Security Considerations](#security-considerations)
10. [Future Improvements](#future-improvements)
11. [Usage Instructions](#usage-instructions)
12. [Conclusion](#conclusion)

## Project Overview

This system leverages both ear canal echo and voice/phrase audio for biometric authentication. It is designed for research and experimental comparison of single-modality and multi-modal (fused) approaches, with robust support for data collection, feature extraction, model training, and prediction.

**Key Features:**
- Multi-modal: Echo (ear canal) and voice (phrase) biometrics (NOT ear-only)
- Flexible: Echo-only, voice-only, fused (early fusion), and late/hybrid fusion
- Robust: Recursive data handling, explicit file/metadata saving, and error handling
- Research-ready: CLI and code support for experimental comparison and ablation

## System Architecture

```
ear_biometrics_prototype/
├── recordings/
│   ├── echo/         # Echo (ear canal) recordings
│   └── voice/        # Voice/phrase recordings
├── data_collector.py # Echo data collection (saves to echo/)
├── phrase_collector.py # Voice data collection (saves to voice/)
├── enhanced_features.py # Multi-modal feature extraction
├── enhanced_train_model.py # Multi-modal model training
├── enhanced_predict.py # Multi-modal prediction
├── utils/
│   ├── audio.py      # Audio I/O and saving utilities
│   └── features.py   # Feature extraction helpers
├── performance_analysis/ # Metrics, confusion matrices, plots
├── requirements.txt
└── PROJECT_DOCUMENTATION.md
```

## Data Collection Process

- **Echo data**: Collected via `data_collector.py`, saved in `recordings/echo/` with metadata JSON.
- **Voice data**: Collected via `phrase_collector.py`, saved in `recordings/voice/` with metadata JSON.
- **Robust path handling**: All parent directories are created as needed; filenames are unique and metadata is always saved.
- **Device selection**: User can select input/output devices; channel handling is robust.
- **Metadata**: Includes user ID, timestamp, device info, and recording parameters.

## EchoID TWS Hardware & Signal Compliance

This project now fully implements the EchoID TWS guide best practices:
- **Chirp-based echo capture**: Linear chirp (500–4000 Hz, 0.5s) for robust ear canal resonance.
- **Echo alignment**: All echo signals are aligned using cross-correlation to compensate for Bluetooth delay.
- **Echo tail checks**: Only samples with ≥300ms usable echo tail after alignment are accepted.
- **Lag offset and centroid shift**: Lag offset (ms) and spectral centroid shift are computed and logged for every echo sample.
- **QA logging**: All quality metrics (RMS, centroid, SNR, lag, tail length, centroid shift) are logged per sample.
- **Strict echo/voice separation**: All echo-specific checks and features are applied only to echo, never to voice.
- **Device recommendations**: In-ear TWS earbuds with in-canal mics are required for best results; open-fit/inline-mic devices are discouraged.

## Data Collection Process (Updated)
- **Echo samples**: Collected with TWS earbuds, chirp tone, and robust QA. Echo tail is aligned and checked for length and quality.
- **QA log**: Each sample logs RMS, centroid, SNR, lag offset, echo tail length, and centroid shift. Bad samples are auto-discarded.
- **Voice samples**: Collected and processed separately, with no echo-specific checks applied.

## Feature Engineering

- **Spectral entropy/flux**: Added as new features for voice (captures randomness and sharpness).
- **Advanced MFCCs**: 20 coefficients, with mean, std, skew, kurtosis, deltas, delta2s
- **Formant, pitch, jitter, HNR**: Already present for voice.
- **Multi-modal separation**: All features are extracted for both echo and voice, with `echo_` and `voice_` prefixes.

## Multi-Modal Feature Extraction & Fusion

- **Feature extraction**: For every sample, both echo and voice features are extracted and saved in the feature matrix.
- **Feature naming**: Features are prefixed with `echo_` or `voice_` for clarity and separation.
- **Inspection**: Easily inspect all `voice_` or `echo_` features in the DataFrame.
- **Fusion strategies**:
  - **Echo-only**: Use only `echo_` features
  - **Voice-only**: Use only `voice_` features
  - **Fused (early fusion)**: Concatenate all features
  - **Late/Hybrid fusion**: Train separate models for each modality, fuse predictions at decision level

## Model Development & Fusion Modes

### Original Model
- Single algorithm: Random Forest
- Basic hyperparameter tuning
- Simple train/test split
- No feature selection or ensemble

### Enhanced Model
1. **Multiple Algorithms**
   - SVM (Best performer)
   - Random Forest
   - Neural Network
   - Gradient Boosting
   - K-Nearest Neighbors
2. **Advanced Training Pipeline**
   - Feature selection (RFE)
   - Standardization
   - Cross-validation
   - Grid search optimization
   - Ensemble methods
3. **Fusion Modes**
   - Echo-only, voice-only, fused (early fusion), late/hybrid fusion
   - CLI: `--feature_mode echo|voice|fused|late_fusion`

### Model Comparison Table (Best Results)

| Algorithm        | Modality      | CV Score | Test Accuracy | Best Parameters                |
|-----------------|--------------|----------|---------------|-------------------------------|
| SVM             | Echo-only    | 78%      | 78%           | C=0.1, kernel=linear           |
| SVM             | Voice-only   | 91%      | 91%           | C=0.1, kernel=linear           |
| SVM             | Fused        | 84%      | 84%           | C=0.1, kernel=linear           |
| Neural Network  | Fused        | 84%      | 84%           | (100,50) hidden layers         |
| Random Forest   | Fused        | 81%      | 81%           | n_estimators=300, max_depth=10 |
| Gradient Boost  | Fused        | 80%      | 80%           | lr=0.1, depth=3                |
| K-NN            | Fused        | 78%      | 78%           | k=5, distance weights          |

- **Late/Hybrid Fusion**: Separate echo and voice models are trained; predictions are fused (e.g., majority vote or confidence-based). Results are similar or slightly better than early fusion, depending on the user and data.

## Performance Analysis

### Overall Performance Comparison

| Metric              | Original Model | Enhanced Model (Best) | Improvement         |
|---------------------|---------------|----------------------|---------------------|
| Overall Accuracy    | 75%           | 91% (voice-only SVM) | +16 pts (+21%)      |
| Number of Features  | 36            | 217 → 80 (selected)  | 6x increase, pruned |
| Feature Types       | Basic MFCC    | Multi-domain, multi-modal | Major enhancement |
| Model Type          | Random Forest | SVM, NN, Ensemble    | Advanced pipeline   |
| Cross-validation    | Basic split   | 5-fold CV, grid search | Robust validation |

### Per-User Performance (Voice-only SVM, Best)

| User      | F1-Score |
|-----------|----------|
| Abhi      | 0.92     |
| Ajmal     | 0.91     |
| Deepak.S  | 0.83     |
| Dhanush.P | 1.00     |
| Hassan    | 0.91     |
| lohith    | 0.92     |

- **Echo-only**: Lower, but still usable (F1: 0.78)
- **Fused**: 0.84 (Neural Net), 0.84 (SVM)
- **Late/Hybrid Fusion**: Similar or slightly better than early fusion

### Confusion Matrix Analysis
- Enhanced system shows much better separation of user classes, especially for voice and fused models.
- Echo-only model has more confusion between similar users.
- Fused and late fusion models reduce false positives/negatives.

## Deep Learning Experiments: Summary and Outcome (2025-06-10)

- Multiple deep learning approaches (CNN, LSTM) were implemented and tested on mel-spectrograms of both echo and voice modalities.
- Despite extensive tuning (augmentation, batch normalization, early stopping, model depth), test accuracy remained very low (typically <25%).
- Data/label integrity checks and visualizations did not reveal obvious errors, but results suggest the current dataset, label extraction, or spectrogram pipeline is not suitable for deep learning.
- **Conclusion:** Deep learning is not effective for this dataset/modality at this time. The project will focus on classical ML models, which have shown more robust and reproducible results.
- Deep learning scripts are retained for reference but are not recommended for use unless the dataset or pipeline is significantly improved.

## Technical Implementation

### Feature Extraction Process
```python
# Multi-modal feature extraction pipeline
1. Audio loading and preprocessing
2. **Adaptive noise reduction (spectral gating or ML-based denoising)**
3. MFCC extraction (20 coefficients)
4. Spectral analysis
5. Resonance frequency detection
6. Wavelet decomposition
7. Statistical calculations
8. Feature selection (RFE)
9. Feature scaling
# Features are separated into echo_ and voice_ branches
```

### Model Training Pipeline
```python
# Enhanced training process
1. Data loading and preprocessing
2. Feature extraction (multi-modal, echo/voice separation)
3. Feature selection (RFE, RFECV, or SelectKBest; CLI: --feature_selection, --num_features)
4. Train/test split (stratified)
5. Feature scaling
6. Model training with grid search (expanded SVM/NN grids)
7. Cross-validation
8. Model comparison
9. Ensemble creation
10. Final evaluation
# Use --feature_mode to select echo, voice, fused, or late_fusion
# Use --feature_selection and --num_features to tune feature selection
```

### Prediction Process
```python
# Enhanced prediction pipeline
1. Audio loading
2. Multi-modal feature extraction (echo/voice separation)
3. Feature selection
4. Feature scaling
5. Model prediction (echo, voice, fused, or late fusion)
6. Confidence calculation
7. Result validation
```

## Security Considerations

- **High accuracy**: Lower false acceptance/rejection
- **Biometric security**: Unique ear canal and voice patterns, hard to spoof
- **Multi-factor ready**: Can combine with other biometrics
- **Template protection**: Encrypt feature templates, secure storage
- **Anti-spoofing**: Liveness detection, replay attack prevention
- **Privacy**: Local processing, minimal data transmission

## Future Improvements

- **Liveness detection**: Add anti-spoofing checks
- **Signal simulation**: Simulate ear/voice signals for augmentation
- **Deep learning**: CNN/RNN for further accuracy
- **Advanced fusion**: Weighted/stacked fusion, meta-learners
- **Real-world robustness**: Device/position invariance, health monitoring

## Usage Instructions

#### Training
```bash
# Train with fused (default, early fusion)
python enhanced_train_model.py
# Train with echo-only
python enhanced_train_model.py --feature_mode echo
# Train with voice-only
python enhanced_train_model.py --feature_mode voice
# Train with late/hybrid fusion
python enhanced_train_model.py --feature_mode late_fusion
# Train with RFECV feature selection
python enhanced_train_model.py --feature_mode voice --feature_selection rfecv
# Train with SelectKBest (univariate) and 30 features
python enhanced_train_model.py --feature_mode voice --feature_selection univariate --num_features 30
# Train with RFE and 80 features
python enhanced_train_model.py --feature_mode voice --feature_selection rfe --num_features 80
```

### Prediction
```bash
# Predict (auto-detects fusion mode from model)
python enhanced_predict.py recordings/audio_file.wav
# Specify fusion mode
python enhanced_predict.py recordings/audio_file.wav --feature_mode fused
python enhanced_predict.py recordings/audio_file.wav --feature_mode echo
python enhanced_predict.py recordings/audio_file.wav --feature_mode voice
python enhanced_predict.py recordings/audio_file.wav --feature_mode late_fusion
```

### Feature Extraction
```bash
# Extract features for all data
python enhanced_features.py
```

## Test-Time Voting via CLI

Test-time voting is now fully supported via the CLI in `enhanced_train_model.py`. This allows robust authentication by aggregating predictions over multiple consecutive samples.

### Usage

Add the following arguments to your CLI command:

- `--voting` : Enable prediction with voting over multiple samples (overrides normal evaluation)
- `--voting_window N` : Number of consecutive samples to use for voting (default: 3)
- `--voting_type {majority,soft}` : Voting method: majority (hard) or soft (probability average, default: majority)
- `--voting_input PATH` : Optional CSV file with features for voting prediction (if not provided, uses the test set)

### Example

```sh
python enhanced_train_model.py --feature_mode voice --voting --voting_window 5 --voting_type soft
```

This will run prediction with soft voting over windows of 5 samples each, using the best model found during training.

See the CLI help (`-h` or `--help`) for more details and examples.

## Conclusion

The system now supports robust, multi-modal authentication using both ear canal echo and voice. All fusion modes (echo-only, voice-only, fused, late/hybrid fusion) are implemented and validated. The architecture is research-ready, with strong performance, flexible experimentation, and clear documentation. Future improvements can further enhance security, accuracy, and real-world robustness.

**Note:** This project is NOT ear-only. Both ear canal echo and voice/phrase modalities are required for best performance and security.

- Hyperparameter tuning: The training pipeline now uses expanded hyperparameter grids for all models (SVM, Random Forest, Gradient Boosting, KNN, MLP) and supports both GridSearchCV (for best accuracy) and RandomizedSearchCV (for faster prototyping). This enables more robust model selection and improved accuracy, with the option to trade off speed and thoroughness as needed.
- Ensembling: The system continues to use late/hybrid fusion and ensemble voting for maximum accuracy and robustness.

## Hardware Requirements for Reliable Echo Biometrics (2025-06 Update)

#### Key Finding
- **Standard consumer earbuds, headphones (wired or wireless), and built-in laptop/PC microphones are NOT suitable for true ear canal echo capture.**
- To reliably capture in-ear echoes, you need specialized hardware with a microphone physically inside the ear canal.

#### Suitable Hardware Options
- **Custom research earbuds** with in-canal mic and speaker (used in academic studies; not commercially available).
- **Medical digital otoscopes** with audio/mic capability (can be adapted for echo experiments).
- **Advanced hearing aids or custom in-ear monitors (IEMs)** with in-canal microphones (may require manufacturer or audiologist collaboration).
- **DIY prototypes** using MEMS microphones and small speakers mounted on custom ear molds (requires electronics and fabrication skills).

#### Not Suitable
- Consumer Bluetooth TWS earbuds (Boult, Muff, etc.)
- Wired headphones with inline mics
- Built-in laptop microphones

#### Why?
- Microphones are not positioned inside the canal, so they cannot capture the true in-ear echo response.
- Bluetooth/Windows limitations further reduce quality and echo capture capability.

#### Recommendation
- For research or deployment, collaborate with academic labs, audiology professionals, or medical device suppliers to access or build in-canal mic hardware.
- For most users, only voice biometrics or other modalities are practical with standard audio devices.

---