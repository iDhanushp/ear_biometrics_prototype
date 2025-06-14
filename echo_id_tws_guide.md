**EchoID: TWS-Based Ear Canal Biometric Authentication - Hardware & Signal Sync Guide**

> **Note:** All EchoID TWS hardware and signal best practices described here are now fully implemented in both data collection and feature extraction. Echo and voice features are strictly separated throughout the pipeline.

---

### 🌍 Overview

EchoID is a contactless biometric authentication system that uses ear canal echo patterns and voiceprint fusion to uniquely identify users. This document focuses specifically on the hardware selection, signal acquisition, and synchronization strategies when using **True Wireless Stereo (TWS)** earbuds.

---

## 🎧 Why TWS is Ideal for EchoID

| Factor                 | TWS Advantage                                                |
| ---------------------- | ------------------------------------------------------------ |
| **Mic Position**       | Mic is close to the ear canal entrance                       |
| **Acoustic Isolation** | In-ear design seals the canal, enhancing echo resonance      |
| **User Comfort**       | No wires, more natural for everyday use                      |
| **Dual Mic Configs**   | Many TWS have internal & external mics for signal processing |
| **Discreetness**       | Looks like regular headphone usage                           |

> Wired earphones place the mic near the mouth (inline), which significantly reduces echo fidelity. TWS earbuds solve this problem by localizing the mic near the source of the echo.

---

## 👤 TWS Device Selection Criteria

### ✅ Minimum Requirements:

- In-ear TWS design (not open fit)
- Mic located inside or near bud (not stem-only)
- Support for mono mode (can use one earbud only)
- Decent SNR with in-canal mic
- Low audio hardware distortion (at least 16-bit audio support)

### ✨ Recommended Models (Tested or Suggested):

- **OnePlus Buds V** (used by EchoID dev)
- **Realme Buds Q2 / Q2s**
- **boAt Airdopes 441 / 381** (sealed fit)
- **Oppo Enco Buds2**
- **Noise Buds VS104**

> Avoid AirPods (standard open type) or non-sealing earbuds, as they do not create the proper echo chamber in the ear canal.

---

## 🔹 Echo Capture Strategy with TWS

### ▶þ Tone Design:

- **Signal**: Linear chirp (500 Hz → 4000 Hz)
- **Duration**: 0.5 sec
- **Sampling Rate**: 44100 Hz (or 48000 Hz for broader frequency resolution)

### ▶þ Recording Design:

- **Start microphone first**: Begin recording 0.2s before tone playback
- **Record duration**: 1.2s (0.5s tone + 0.7s echo tail)
- **Record mono audio**

### ▶þ Sync Method (Critical!):

Use **cross-correlation** to align the recorded signal with the played chirp.

```python
from scipy.signal import correlate

def align_echo(tone, recorded):
    corr = correlate(recorded, tone, mode='full')
    lag = np.argmax(corr) - len(tone)
    aligned_tail = recorded[lag + len(tone):]
    return aligned_tail
```

This method compensates for unknown Bluetooth playback delays.

---

## 📊 Quality Control and Signal Validation

### ✅ Auto-discard bad samples:

- **RMS Threshold**: < 0.0005 = too quiet
- **Spectral Centroid Shift**: Should change post-tone
- **Echo Tail Length**: At least 300ms usable data

### ▶þ Log Per-Sample Metrics:

- RMS
- Spectral centroid
- Peak-to-noise ratio
- Lag offset (from cross-correlation)

---

## 🧹 Future Enhancements

- Use binaural recording (left and right TWS echo differences)
- Add inaudible signal modes (>18 kHz) for passive continuous auth
- Delay calibration per device (learn TWS latency on first install)

---

## 📖 Summary

Using TWS earbuds for EchoID echo capture is the most viable strategy for consistent biometric signal quality. With cross-correlation-based alignment, RMS gating, and proper device selection, you can achieve robust ear canal echo capture across a range of consumer devices.

> This approach allows EchoID to scale to mobile devices while maintaining authentication accuracy, stealth, and comfort.

