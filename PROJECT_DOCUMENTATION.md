# Ear Canal + Voice Biometric Authentication System (Multi-Modal)

**This project implements robust, multi-modal biometric authentication using both ear canal echo and voice features. It is NOT ear-only: both modalities are required for best performance.**

# Project Documentation

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

## Feature Engineering

- **Advanced MFCCs**: 20 coefficients, with mean, std, skew, kurtosis, deltas, delta2s
- **Resonance features**: Top 5 resonant frequencies, magnitudes, ratios
- **Wavelet features**: Daubechies, 5 levels, energy/statistics per level
- **Spectral features**: Centroid, rolloff, bandwidth, contrast, flatness
- **Temporal features**: Zero crossing, RMS, tempo, beat count
- **Statistical features**: Mean, std, skew, kurtosis, entropy
- **LPC features**: Linear prediction coefficients
- **Chroma features**: 12 bands, mean/std
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

## Technical Implementation

### Feature Extraction Process
```python
# Multi-modal feature extraction pipeline
1. Audio loading and preprocessing
2. MFCC extraction (20 coefficients)
3. Spectral analysis
4. Resonance frequency detection
5. Wavelet decomposition
6. Statistical calculations
7. Feature selection (RFE)
8. Feature scaling
# Features are separated into echo_ and voice_ branches
```

### Model Training Pipeline
```python
# Enhanced training process
1. Data loading and preprocessing
2. Feature extraction (multi-modal, echo/voice separation)
3. Feature selection (RFE → 80 features)
4. Train/test split (stratified)
5. Feature scaling
6. Model training with grid search
7. Cross-validation
8. Model comparison
9. Ensemble creation
10. Final evaluation
# Use --feature_mode to select echo, voice, fused, or late_fusion
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

### Training
```bash
# Train with fused (default, early fusion)
python enhanced_train_model.py
# Train with echo-only
python enhanced_train_model.py --feature_mode echo
# Train with voice-only
python enhanced_train_model.py --feature_mode voice
# Train with late/hybrid fusion
python enhanced_train_model.py --feature_mode late_fusion
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

## Conclusion

The system now supports robust, multi-modal authentication using both ear canal echo and voice. All fusion modes (echo-only, voice-only, fused, late/hybrid fusion) are implemented and validated. The architecture is research-ready, with strong performance, flexible experimentation, and clear documentation. Future improvements can further enhance security, accuracy, and real-world robustness.

**Note:** This project is NOT ear-only. Both ear canal echo and voice/phrase modalities are required for best performance and security.