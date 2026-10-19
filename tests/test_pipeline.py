"""
Unit tests for the Radar Signal Classifier pipeline.

Tests cover signal generation, noise injection, feature extraction,
and model prediction to ensure pipeline integrity.
"""

import os
import sys
import numpy as np
import pytest

# Ensure src is importable
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC_DIR)

from src.config import SAMPLE_RATE, N_SAMPLES, DURATION, TIME
from src.signal_generator import (
    generate_sine_wave,
    generate_square_wave,
    generate_chirp_signal,
    generate_noisy_signal,
)
from src.noise import add_gaussian_noise, add_white_noise, add_noise
from src.feature_extractor import (
    extract_features,
    extract_mean,
    extract_std,
    extract_rms,
    extract_energy,
    extract_peak_to_peak,
    extract_dominant_frequency,
)


# ─── Signal Generation Tests ───────────────────────────────────────────────

class TestSignalGeneration:
    """Tests for signal_generator.py"""

    def test_sine_wave_shape(self):
        signal = generate_sine_wave()
        assert len(signal) == N_SAMPLES, f"Expected {N_SAMPLES} samples"

    def test_sine_wave_amplitude(self):
        amp = 2.0
        signal = generate_sine_wave(freq=5.0, amplitude=amp)
        assert np.max(np.abs(signal)) <= amp + 1e-10

    def test_square_wave_shape(self):
        signal = generate_square_wave()
        assert len(signal) == N_SAMPLES

    def test_square_wave_values(self):
        """Square wave should only have values at +amp and -amp."""
        amp = 1.0
        signal = generate_square_wave(freq=5.0, amplitude=amp)
        unique_vals = np.unique(signal)
        assert len(unique_vals) == 2
        assert np.isclose(unique_vals[0], -amp)
        assert np.isclose(unique_vals[1], amp)

    def test_chirp_signal_shape(self):
        signal = generate_chirp_signal()
        assert len(signal) == N_SAMPLES

    def test_chirp_signal_bounded(self):
        amp = 1.5
        signal = generate_chirp_signal(amplitude=amp)
        assert np.max(np.abs(signal)) <= amp + 1e-10

    def test_noisy_signal_shape(self):
        signal = generate_noisy_signal()
        assert len(signal) == N_SAMPLES

    def test_noisy_signal_has_noise(self):
        """Noisy signal should differ from the base sine wave."""
        np.random.seed(42)
        sine = generate_sine_wave(freq=5.0, amplitude=1.0)
        np.random.seed(42)
        noisy = generate_noisy_signal(freq=5.0, amplitude=1.0, noise_level=1.5)
        assert not np.allclose(sine, noisy)


# ─── Noise Injection Tests ─────────────────────────────────────────────────

class TestNoiseInjection:
    """Tests for noise.py"""

    def test_gaussian_noise_shape(self):
        signal = np.zeros(N_SAMPLES)
        noisy = add_gaussian_noise(signal, noise_level=0.5)
        assert len(noisy) == N_SAMPLES

    def test_gaussian_noise_adds_variation(self):
        signal = np.ones(N_SAMPLES)
        noisy = add_gaussian_noise(signal, noise_level=0.5)
        assert np.std(noisy) > 0

    def test_white_noise_shape(self):
        signal = np.zeros(N_SAMPLES)
        noisy = add_white_noise(signal, noise_level=0.5)
        assert len(noisy) == N_SAMPLES

    def test_white_noise_bounded(self):
        signal = np.zeros(N_SAMPLES)
        noisy = add_white_noise(signal, noise_level=0.5)
        assert np.all(noisy >= -0.5)
        assert np.all(noisy <= 0.5)

    def test_add_noise_dispatcher(self):
        signal = np.zeros(N_SAMPLES)
        g = add_noise(signal, "gaussian", 0.5)
        w = add_noise(signal, "white", 0.5)
        assert len(g) == N_SAMPLES
        assert len(w) == N_SAMPLES

    def test_add_noise_invalid_type(self):
        signal = np.zeros(N_SAMPLES)
        with pytest.raises(ValueError):
            add_noise(signal, "invalid_type", 0.5)


# ─── Feature Extraction Tests ──────────────────────────────────────────────

class TestFeatureExtraction:
    """Tests for feature_extractor.py"""

    def test_extract_features_keys(self):
        signal = generate_sine_wave()
        features = extract_features(signal, SAMPLE_RATE)
        expected_keys = {
            "mean", "std", "max", "min", "rms",
            "energy", "peak_to_peak", "dominant_frequency",
        }
        assert set(features.keys()) == expected_keys

    def test_extract_features_count(self):
        signal = generate_sine_wave()
        features = extract_features(signal, SAMPLE_RATE)
        assert len(features) == 8

    def test_sine_mean_near_zero(self):
        signal = generate_sine_wave(freq=5.0, amplitude=1.0)
        assert abs(extract_mean(signal)) < 0.1

    def test_sine_rms(self):
        """RMS of a sine wave should be amp / sqrt(2)."""
        amp = 1.0
        signal = generate_sine_wave(freq=5.0, amplitude=amp)
        expected_rms = amp / np.sqrt(2)
        assert abs(extract_rms(signal) - expected_rms) < 0.05

    def test_energy_positive(self):
        signal = generate_sine_wave()
        assert extract_energy(signal) > 0

    def test_peak_to_peak(self):
        amp = 1.0
        signal = generate_sine_wave(freq=5.0, amplitude=amp)
        ptp = extract_peak_to_peak(signal)
        assert abs(ptp - 2.0 * amp) < 0.05

    def test_dominant_frequency_sine(self):
        """Dominant frequency of a 10 Hz sine should be ~10."""
        signal = generate_sine_wave(freq=10.0, amplitude=1.0)
        dom_freq = extract_dominant_frequency(signal, SAMPLE_RATE)
        assert abs(dom_freq - 10.0) < 2.0

    def test_features_differ_across_classes(self):
        """Different signal types should produce different feature vectors."""
        sine = generate_sine_wave(freq=5.0)
        square = generate_square_wave(freq=5.0)
        chirp = generate_chirp_signal()

        f_sine = extract_features(sine, SAMPLE_RATE)
        f_square = extract_features(square, SAMPLE_RATE)
        f_chirp = extract_features(chirp, SAMPLE_RATE)

        # At least energy should differ
        assert f_sine["energy"] != f_square["energy"]
        assert f_sine["dominant_frequency"] != f_chirp["dominant_frequency"] or \
               f_sine["energy"] != f_chirp["energy"]


# ─── Predictor Tests ───────────────────────────────────────────────────────

class TestPredictor:
    """Tests for predictor.py — requires trained model artifacts."""

    @pytest.fixture
    def rf_predictor(self):
        from src.predictor import RadarPredictor
        model_path = os.path.join(SRC_DIR, "..", "results", "best_model.pkl")
        if not os.path.exists(model_path):
            pytest.skip("Trained model not found — run train_model.py first")
        return RadarPredictor(model_type="rf")

    def test_rf_sine_prediction(self, rf_predictor):
        signal = generate_sine_wave(freq=5.0, amplitude=1.0)
        result = rf_predictor.predict(signal)
        assert result["label"] == "sine"
        assert result["confidence"] > 50.0

    def test_rf_square_prediction(self, rf_predictor):
        signal = generate_square_wave(freq=5.0, amplitude=1.0)
        result = rf_predictor.predict(signal)
        assert result["label"] == "square"

    def test_rf_chirp_prediction(self, rf_predictor):
        signal = generate_chirp_signal(f0=2.0, f1=50.0, amplitude=1.0)
        result = rf_predictor.predict(signal)
        assert result["label"] == "chirp"

    def test_rf_result_structure(self, rf_predictor):
        signal = generate_sine_wave()
        result = rf_predictor.predict(signal)
        assert "label" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "model_type" in result
        assert isinstance(result["probabilities"], dict)

    def test_rf_wrong_signal_length(self, rf_predictor):
        signal = np.zeros(100)  # Wrong length
        with pytest.raises(ValueError):
            rf_predictor.predict(signal)

    def test_rf_batch_prediction(self, rf_predictor):
        signals = [
            generate_sine_wave(),
            generate_square_wave(),
            generate_chirp_signal(),
        ]
        results = rf_predictor.predict_batch(signals)
        assert len(results) == 3
        assert all("label" in r for r in results)


# ─── Config Tests ──────────────────────────────────────────────────────────

class TestConfig:
    """Tests for config.py"""

    def test_sample_rate(self):
        assert SAMPLE_RATE == 500

    def test_n_samples(self):
        assert N_SAMPLES == int(SAMPLE_RATE * DURATION)

    def test_time_array_length(self):
        assert len(TIME) == N_SAMPLES

    def test_time_array_range(self):
        assert TIME[0] == 0.0
        assert TIME[-1] < DURATION


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
