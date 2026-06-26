import numpy as np
from scipy.fft import fft, fftfreq
from config import SAMPLE_RATE


def extract_mean(signal):
    return np.mean(signal)


def extract_std(signal):
    return np.std(signal)


def extract_max(signal):
    return np.max(signal)


def extract_min(signal):
    return np.min(signal)


def extract_rms(signal):
    """Root Mean Square — overall signal magnitude, regardless of sign."""
    return np.sqrt(np.mean(signal ** 2))


def extract_energy(signal):
    """Signal power/energy — sum of squared values."""
    return np.sum(signal ** 2)


def extract_peak_to_peak(signal):
    """Difference between max and min — how much the signal swings."""
    return np.max(signal) - np.min(signal)


def extract_dominant_frequency(signal, sample_rate=500):
    """
    Uses FFT to find which frequency has the most energy in the signal.
    This is the single most informative feature for telling sine/square/chirp apart.
    """
    n = len(signal)
    fft_values = fft(signal)
    fft_freqs = fftfreq(n, d=1 / sample_rate)

    # Only look at the positive-frequency half (FFT output is mirrored)
    half = n // 2
    magnitudes = np.abs(fft_values[:half])
    freqs = fft_freqs[:half]

    # Ignore frequency 0 (the DC/average component) when finding the peak
    if len(magnitudes) > 1:
        dominant_idx = np.argmax(magnitudes[1:]) + 1
    else:
        dominant_idx = 0

    return freqs[dominant_idx]


def extract_features(signal, sample_rate=500):
    """
    Builds the full feature vector for one signal, matching the spec:
    [mean, std, max, min, rms, energy, dominant_frequency]
    """
    return {
        "mean": extract_mean(signal),
        "std": extract_std(signal),
        "max": extract_max(signal),
        "min": extract_min(signal),
        "rms": extract_rms(signal),
        "energy": extract_energy(signal),
        "peak_to_peak": extract_peak_to_peak(signal),
        "dominant_frequency": extract_dominant_frequency(signal, sample_rate),
    }


if __name__ == "__main__":
    import os
    import sys
    import matplotlib.pyplot as plt

    sys.path.append(os.path.dirname(__file__))
    from signal_generator import generate_sine_wave, generate_chirp_signal, TIME, SAMPLE_RATE

    sine = generate_sine_wave(freq=10.0)
    chirp_sig = generate_chirp_signal()

    sine_features = extract_features(sine, SAMPLE_RATE)
    chirp_features = extract_features(chirp_sig, SAMPLE_RATE)

    print("Sine wave features:")
    for k, v in sine_features.items():
        print(f"  {k}: {v:.4f}")

    print("\nChirp signal features:")
    for k, v in chirp_features.items():
        print(f"  {k}: {v:.4f}")

    # Visualize the FFT of the sine wave to confirm dominant frequency detection works
    from scipy.fft import fft, fftfreq
    n = len(sine)
    fft_vals = np.abs(fft(sine))[:n // 2]
    freqs = fftfreq(n, d=1 / SAMPLE_RATE)[:n // 2]

    plt.figure(figsize=(8, 4))
    plt.plot(freqs, fft_vals)
    plt.title("FFT of Sine Wave (freq=10 Hz) — peak should be near 10")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 50)

    output_path = os.path.join("..", "results", "fft_check.png")
    plt.savefig(output_path)
    print(f"\nSaved FFT plot to {output_path}")