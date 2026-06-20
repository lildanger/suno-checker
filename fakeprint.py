"""
Standalone fakeprint extraction for AI music detection.
Pure numpy/scipy/librosa — no torch needed. CPU-only, lightweight.

Based on: "A Fourier Explanation of AI-Music Artifacts" (ISMIR 2025)
Original: https://github.com/lofcz/ai-music-detector
"""
import numpy as np
from scipy.ndimage import minimum_filter1d


# -- Config (matching lofcz pretrained model) --
SAMPLE_RATE = 16000
N_FFT = 8192
FREQ_MIN = 1000   # Hz
FREQ_MAX = 8000   # Hz
HULL_AREA = 10     # bins for lower hull
MAX_DB = 5.0       # dB clipping
MIN_DB = -45.0     # dB floor
MAX_DURATION = 300  # seconds, truncate longer audio


def extract_fakeprint(audio_path: str) -> np.ndarray:
    """
    Extract 3585-dim fakeprint vector from an audio file.

    Returns:
        np.ndarray of shape (3585,) — normalized [0, 1]
    """
    import librosa

    # 1. Load and resample to 16kHz mono
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    # 2. Truncate if too long
    max_samples = MAX_DURATION * SAMPLE_RATE
    if len(y) > max_samples:
        y = y[:max_samples]

    # 3. STFT (power spectrum)
    D = librosa.stft(y, n_fft=N_FFT, hop_length=N_FFT//4, center=False)
    power = np.abs(D) ** 2

    # 4. Convert to dB and average across time
    power_db = 10 * np.log10(np.maximum(power, 1e-10))
    mean_spectrum = power_db.mean(axis=1)  # average over time

    # 5. Apply frequency mask (1000-8000 Hz)
    freq_bins = np.linspace(0, sr / 2, num=(N_FFT // 2) + 1)
    mask = (freq_bins >= FREQ_MIN) & (freq_bins <= FREQ_MAX)
    freq_spectrum = mean_spectrum[mask]

    # 6. Compute lower hull via minimum filter
    hull = minimum_filter1d(freq_spectrum, size=HULL_AREA, mode='nearest')
    hull = np.clip(hull, MIN_DB, None)

    # 7. Residue = spectrum - hull (the "fakeprint" peaks)
    residue = np.clip(freq_spectrum - hull, 0, None)

    # 8. Clip and normalize to [0, 1]
    residue = np.clip(residue, 0, MAX_DB)
    fakeprint = residue / (np.max(residue) + 1e-6)

    return fakeprint.astype(np.float32)


def predict_ai_probability(audio_path: str, onnx_model_path: str) -> float:
    """
    Full pipeline: extract fakeprint + ONNX inference.

    Returns:
        float 0.0 (real) to 1.0 (AI-generated)
    """
    import onnxruntime as ort

    fakeprint = extract_fakeprint(audio_path)
    session = ort.InferenceSession(onnx_model_path)
    output = session.run(None, {"fakeprint": fakeprint.reshape(1, -1).astype(np.float32)})
    return float(output[0][0, 0])


# -- Self-test --
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        fp = extract_fakeprint(path)
        print(f"File: {path}")
        print(f"Fakeprint shape: {fp.shape}")
        print(f"Fakeprint range: [{fp.min():.4f}, {fp.max():.4f}]")
    else:
        # Dry-run with random noise
        y = np.random.randn(SAMPLE_RATE * 10).astype(np.float32)
        import librosa
        D = librosa.stft(y, n_fft=N_FFT, hop_length=N_FFT//4, center=False)
        power = np.abs(D) ** 2
        power_db = 10 * np.log10(np.maximum(power, 1e-10))
        mean_spec = power_db.mean(axis=1)
        freq_bins = np.linspace(0, SAMPLE_RATE/2, num=(N_FFT//2)+1)
        mask = (freq_bins >= FREQ_MIN) & (freq_bins <= FREQ_MAX)
        freq_spec = mean_spec[mask]
        hull = minimum_filter1d(freq_spec, size=HULL_AREA, mode='nearest')
        hull = np.clip(hull, MIN_DB, None)
        residue = np.clip(freq_spec - hull, 0, None)
        residue = np.clip(residue, 0, MAX_DB)
        fp = residue / (np.max(residue) + 1e-6)
        print(f"Dry-run OK. Fakeprint dim: {len(fp)}")
