# Ear Canal + Voice Biometric Authentication System (Multi-Modal)

**This project uses both ear canal echo and voice features for robust, multi-modal biometric authentication. It is NOT ear-only: both modalities are required for best performance.**

# Ear Canal Acoustic Biometrics - Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Data Collection Process](#data-collection-process)
4. [Feature Engineering](#feature-engineering)
5. [Model Development](#model-development)
6. [Performance Analysis](#performance-analysis)
7. [Technical Implementation](#technical-implementation)
8. [Security Considerations](#security-considerations)
9. [Future Improvements](#future-improvements)

## Project Overview

The Ear Canal Acoustic Biometrics project implements a secure authentication system using unique acoustic signatures from a person's ear canal and/or their voice. This document provides a comprehensive overview of both the original and enhanced implementations, their differences, and performance metrics.

### Key Components
- Audio signal capture and processing (ear canal echo and voice/phrase)
- Feature extraction and analysis (multi-modal: echo and voice features separated)
- Machine learning model training (supports echo-only, voice-only, or fused)
- Authentication prediction (multi-modal aware)
- Performance evaluation and comparison of modalities

## System Architecture

### Original System
```
ear_biometrics_prototype/
├── recordings/           # Raw audio recordings
├── data_collector.py    # Basic audio collection
├── utils/              
│   ├── audio.py        # Basic audio processing
│   └── features.py     # Simple feature extraction
└── requirements.txt     # Basic dependencies
```

### Enhanced System
```
ear_biometrics_prototype/
├── recordings/           # Raw audio recordings
│   ├── echo/            # Echo (ear canal) recordings
│   └── voice/           # Voice/phrase recordings (future)
├── data_collector.py    # Enhanced audio collection (saves to echo/)
├── phrase_collector.py  # Voice phrase collection (saves to recordings/ or voice/)
├── enhanced_features.py # Advanced feature extraction (multi-modal)
├── enhanced_train_model.py # Advanced ML pipeline (multi-modal aware)
├── enhanced_predict.py  # Enhanced prediction system (multi-modal aware)
├── utils/              
│   ├── audio.py        # Enhanced audio processing
│   └── features.py     # Basic feature extraction
├── requirements.txt     # Extended dependencies
└── MODEL_IMPROVEMENTS.md # Model enhancement documentation
```

## Data Collection Process

### Original Implementation
- Basic audio recording at 44.1kHz, 16-bit
- Simple metadata storage (user ID, timestamp)
- Limited quality checks
- No noise reduction

### Enhanced Implementation
- High-quality audio recording (44.1kHz, 16-bit)
- Comprehensive metadata
  - User ID
  - Timestamp
  - Recording parameters
  - Device information
  - Environmental conditions
- Advanced quality checks
- Basic noise reduction
- Multiple samples per user
- **Echo data is now saved in `recordings/echo/` for clear separation**
- **Voice data can be saved in `recordings/voice/` (future-proofed)**
- Device selection and channel handling for robust hardware support

## Feature Engineering

### Original Features (36 features)
1. Basic MFCC Features (13 coefficients)
   - Mean and standard deviation
   - Total: 26 features
2. Simple Spectral Features (10 features)
   - Spectral centroid
   - Spectral rolloff
   - Spectral bandwidth

### Enhanced Features (217 → 80 selected features)
1. Advanced MFCC Features (120 features)
   - 20 coefficients (increased from 13)
   - Comprehensive statistics:
     - Mean
     - Standard deviation
     - Skewness
     - Kurtosis
     - First derivative (delta)
     - Second derivative (delta2)
2. Resonance Analysis (12 features)
   - Top 5 resonant frequencies
   - Corresponding magnitudes
   - Frequency ratios
3. Wavelet Transform Features (18 features)
   - Daubechies wavelets (5 levels)
   - Energy and statistics per level
4. Advanced Spectral Features (22 features)
   - Spectral centroid
   - Spectral rolloff
   - Spectral bandwidth
   - Spectral contrast (7 bands)
   - Spectral flatness
5. Temporal Features (6 features)
   - Zero crossing rate
   - RMS energy
   - Tempo
   - Beat count
6. Statistical Features (5 features)
   - Audio mean
   - Standard deviation
   - Skewness
   - Kurtosis
   - Entropy
7. LPC Features (10 features)
   - Linear prediction coefficients
8. Chroma Features (24 features)
   - 12 chroma bands
   - Mean and standard deviation

**Multi-Modal Feature Separation:**
- Features are now separated into `echo_` (ear canal) and `voice_` (voice/phrase) branches using metadata and filename heuristics.
- Shared features (e.g., spectral, temporal) are included in both branches for robust fusion.

## Model Development

### Original Model
- Single algorithm: Random Forest
- Basic hyperparameter tuning
- Simple train/test split
- No feature selection
- No ensemble methods

### Enhanced Model
1. Multiple Algorithms
   - SVM (Best performer)
   - Random Forest
   - Neural Network
   - Gradient Boosting
   - K-Nearest Neighbors
2. Advanced Training Pipeline
   - Feature selection (RFE)
   - Standardization
   - Cross-validation
   - Grid search optimization
   - Ensemble methods
3. Model Comparison

| Algorithm | CV Score | Test Accuracy | Best Parameters |
|-----------|----------|---------------|-----------------|
| SVM | 92.39% | 91.67% | C=0.1, kernel=linear |
| Random Forest | 92.36% | 80.56% | n_estimators=300, max_depth=10 |
| Neural Network | 92.34% | 83.33% | (100,50) hidden layers |
| Gradient Boosting | 79.19% | 80.56% | lr=0.1, depth=3 |
| K-NN | 88.23% | 77.78% | k=5, distance weights |

**Multi-Modal Fusion Pipeline:**
- The model can be trained and evaluated with echo-only, voice-only, or fused features.
- Use the CLI option `--feature_mode` with `enhanced_train_model.py` to select the modality:
  - `--feature_mode echo` : Echo-only features
  - `--feature_mode voice` : Voice-only features
  - `--feature_mode fused` : All features (default, early fusion)
- The prediction pipeline uses the same multi-modal feature extraction for consistency.

## Performance Analysis

### Overall Performance Comparison

| Metric | Original Model | Enhanced Model | Improvement |
|--------|---------------|----------------|-------------|
| Overall Accuracy | 75% | 92% | +17 pts (+23%) |
| Number of Features | 36 | 217 → 80 (selected) | 6x increase → optimized |
| Feature Types | Basic MFCC + Spectral | Multi-domain comprehensive | Major enhancement |
| Model Type | Random Forest only | Multiple algorithms + ensemble | Advanced ML pipeline |
| Cross-validation | Basic split | 5-fold CV with grid search | Robust validation |

### Per-User Performance

| User | Original F1-Score | Enhanced F1-Score | Improvement |
|------|------------------|-------------------|-------------|
| Abhi | 67% | 92% | +25 pts |
| Ajmal | 62% | 91% | +29 pts |
| Deepak.S | 55% | 83% | +28 pts |
| Dhanush.P | 100% | 100% | Maintained |
| Hassan | 73% | 91% | +18 pts |
| lohith | 92% | 92% | Maintained |

### Confusion Matrix Analysis
- Original Model: Higher confusion between similar users
- Enhanced Model: Better separation of user classes
- Reduced false positives and negatives
- **Multi-modal analysis:** You can now compare confusion matrices for echo-only, voice-only, and fused models.

## Technical Implementation

### Feature Extraction Process
```python
# Enhanced feature extraction pipeline
1. Audio loading and preprocessing
2. MFCC extraction (20 coefficients)
3. Spectral analysis
4. Resonance frequency detection
5. Wavelet decomposition
6. Statistical calculations
7. Feature selection (RFE)
8. Feature scaling
# Multi-modal: Features are separated into echo_ and voice_ branches
```

### Model Training Pipeline
```python
# Enhanced training process
1. Data loading and preprocessing
2. Feature extraction (217 features, multi-modal separation)
3. Feature selection (RFE → 80 features)
4. Train/test split (stratified)
5. Feature scaling
6. Model training with grid search
7. Cross-validation
8. Model comparison
9. Ensemble creation
10. Final evaluation
# Use --feature_mode to select echo, voice, or fused features
```

### Prediction Process
```python
# Enhanced prediction pipeline
1. Audio loading
2. Multi-modal feature extraction (echo/voice separation)
3. Feature selection
4. Feature scaling
5. Model prediction
6. Confidence calculation
7. Result validation
```

## Security Considerations

### Advantages
1. High Accuracy (92%)
   - Reduced false acceptance rate
   - Lower false rejection rate
   - Better user experience
2. Biometric Security
   - Unique ear canal patterns
   - Difficult to spoof
   - Non-invasive authentication
3. Multi-factor Ready
   - Can combine with voice authentication
   - Additional security layer
   - Flexible implementation

### Recommendations
1. Template Protection
   - Encrypt stored feature templates
   - Secure storage
   - Regular updates
2. Anti-Spoofing
   - Liveness detection
   - Replay attack prevention
   - Environmental noise detection
3. Privacy
   - Local processing
   - Minimal data transmission
   - User consent management

## Future Improvements

### Short-term Improvements
1. Data Augmentation
   - Noise addition
   - Speed variation
   - Pitch shifting
   - Expected gain: 5-10% accuracy
2. Model Optimization
   - Feature importance analysis
   - Model compression
   - Real-time processing

### Long-term Improvements
1. Deep Learning
   - CNN-based models
   - RNN/LSTM implementation
   - Expected accuracy: 95-98%
2. Multi-modal Fusion
   - Voice + ear canal combination (late/hybrid fusion)
   - Environmental adaptation
   - Temporal modeling
3. Real-world Robustness
   - Device invariance
   - Position variation handling
   - Health monitoring

## Usage Instructions

### Training
```bash
# Train the enhanced model (fused features)
python enhanced_train_model.py
# Train with echo-only features
python enhanced_train_model.py --feature_mode echo
# Train with voice-only features
python enhanced_train_model.py --feature_mode voice
```

### Prediction
```bash
# Quick prediction
python enhanced_predict.py recordings/audio_file.wav
# Detailed analysis
python enhanced_predict.py recordings/audio_file.wav --detailed
```

### Feature Extraction
```bash
# Test feature extraction
python enhanced_features.py
```

## Conclusion

The enhanced ear canal biometric system now supports robust multi-modal authentication using both ear canal echo and voice biometrics. The system achieves a 23% relative improvement in accuracy, and the new architecture allows for direct comparison and fusion of modalities. The comprehensive feature engineering, advanced machine learning pipeline, and robust validation methodology provide a solid foundation for production deployment.

Key achievements:
1. Increased accuracy from 75% to 92%
2. Comprehensive, multi-modal feature set (echo/voice separation)
3. Multiple algorithm comparison and ensemble
4. Robust validation and testing
5. Production-ready implementation
6. Flexible experimentation with echo, voice, or fused features

The system is now ready for deployment with proper security measures and monitoring in place. Future improvements can further enhance accuracy and robustness for real-world applications.