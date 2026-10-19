"""
Single-Signal Prediction Utility

Provides a clean interface for classifying individual radar signals using
either the Random Forest (classical ML) or CNN (deep learning) model.

Usage:
    python predictor.py                          # Interactive demo
    python predictor.py --type sine --freq 5.0   # Generate and classify
    python predictor.py --file signal.csv        # Classify from file

This module is also importable for use in the API layer and dashboard.
"""

import os
import sys
import argparse
import numpy as np
import joblib

# Resolve project paths relative to this file's location
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

sys.path.insert(0, SRC_DIR)

from config import SAMPLE_RATE, N_SAMPLES
from signal_generator import (
    generate_sine_wave,
    generate_square_wave,
    generate_chirp_signal,
    generate_noisy_signal,
)
from feature_extractor import extract_features


# Label mapping for CNN (numeric → string)
CNN_LABEL_MAP = {0: "sine", 1: "square", 2: "chirp", 3: "noisy"}
SIGNAL_CLASSES = ["sine", "square", "chirp", "noisy"]


class RadarPredictor:
    """
    Unified predictor for radar signal classification.

    Supports both Random Forest (feature-based) and CNN (raw signal) models.
    Loads model artifacts lazily on first use.
    """

    def __init__(self, model_type="rf"):
        """
        Initialize the predictor.

        Args:
            model_type: 'rf' for Random Forest (default) or 'cnn' for CNN.
        """
        self.model_type = model_type.lower()
        self._rf_model = None
        self._scaler = None
        self._cnn_model = None
        self._device = None

    def _load_rf(self):
        """Load the Random Forest model and scaler."""
        if self._rf_model is None:
            model_path = os.path.join(RESULTS_DIR, "best_model.pkl")
            scaler_path = os.path.join(RESULTS_DIR, "scaler.pkl")

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model not found at {model_path}. "
                    "Run 'python train_model.py' first."
                )
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(
                    f"Scaler not found at {scaler_path}. "
                    "Run 'python train_model.py' first."
                )

            self._rf_model = joblib.load(model_path)
            self._scaler = joblib.load(scaler_path)

    def _load_cnn(self):
        """Load the CNN model."""
        if self._cnn_model is None:
            try:
                import torch
                from cnn_model import SignalCNN
            except ImportError as e:
                raise ImportError(
                    "PyTorch is required for CNN inference. Install it with: pip install torch"
                ) from e

            cnn_path = os.path.join(RESULTS_DIR, "cnn_model.pth")

            if not os.path.exists(cnn_path):
                raise FileNotFoundError(
                    f"CNN model not found at {cnn_path}. "
                    "Run 'python cnn_model.py' first."
                )

            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._cnn_model = SignalCNN(num_classes=4).to(self._device)
            self._cnn_model.load_state_dict(
                torch.load(cnn_path, map_location=self._device, weights_only=True)
            )
            self._cnn_model.eval()

    def predict(self, signal):
        """
        Classify a single signal.

        Args:
            signal: numpy array of shape (N_SAMPLES,) — the raw signal.

        Returns:
            dict with keys:
                - 'label': predicted class name (str)
                - 'confidence': prediction confidence (float, 0-100)
                - 'probabilities': dict of {class_name: probability}
                - 'model_type': which model was used
        """
        signal = np.asarray(signal, dtype=np.float64)

        if len(signal) != N_SAMPLES:
            raise ValueError(
                f"Signal must have {N_SAMPLES} samples, got {len(signal)}."
            )

        if self.model_type == "rf":
            return self._predict_rf(signal)
        elif self.model_type == "cnn":
            return self._predict_cnn(signal)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _predict_rf(self, signal):
        """Predict using Random Forest (feature-based)."""
        self._load_rf()

        # Extract features
        features = extract_features(signal, SAMPLE_RATE)
        feature_values = np.array(list(features.values())).reshape(1, -1)

        # Scale and predict
        features_scaled = self._scaler.transform(feature_values)
        prediction = self._rf_model.predict(features_scaled)[0]
        probabilities = self._rf_model.predict_proba(features_scaled)[0]
        confidence = float(np.max(probabilities) * 100)

        # Build probability dict
        prob_dict = {
            cls: float(prob * 100)
            for cls, prob in zip(self._rf_model.classes_, probabilities)
        }

        return {
            "label": prediction,
            "confidence": confidence,
            "probabilities": prob_dict,
            "model_type": "Random Forest",
            "features": features,
        }

    def _predict_cnn(self, signal):
        """Predict using CNN (raw signal)."""
        import torch
        self._load_cnn()

        # Convert to tensor: (1, 1, N_SAMPLES)
        signal_tensor = (
            torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0).to(self._device)
        )

        with torch.no_grad():
            outputs = self._cnn_model(signal_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_idx = int(np.argmax(probabilities))

        label = CNN_LABEL_MAP[predicted_idx]
        confidence = float(probabilities[predicted_idx] * 100)

        prob_dict = {
            CNN_LABEL_MAP[i]: float(prob * 100)
            for i, prob in enumerate(probabilities)
        }

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": prob_dict,
            "model_type": "CNN",
        }

    def predict_batch(self, signals):
        """
        Classify multiple signals.

        Args:
            signals: list of numpy arrays, each of shape (N_SAMPLES,).

        Returns:
            list of prediction dicts (same format as predict()).
        """
        return [self.predict(signal) for signal in signals]


def generate_signal(signal_type, freq=5.0, amplitude=1.0, **kwargs):
    """
    Helper to generate a signal by type name.

    Args:
        signal_type: one of 'sine', 'square', 'chirp', 'noisy'
        freq: base frequency in Hz
        amplitude: signal amplitude
        **kwargs: additional params (f1 for chirp, noise_level for noisy)

    Returns:
        numpy array of shape (N_SAMPLES,)
    """
    signal_type = signal_type.lower()

    if signal_type == "sine":
        return generate_sine_wave(freq, amplitude)
    elif signal_type == "square":
        return generate_square_wave(freq, amplitude)
    elif signal_type == "chirp":
        f0 = kwargs.get("f0", freq)
        f1 = kwargs.get("f1", 50.0)
        return generate_chirp_signal(f0, f1, amplitude)
    elif signal_type == "noisy":
        noise_level = kwargs.get("noise_level", 1.5)
        return generate_noisy_signal(freq, amplitude, noise_level)
    else:
        raise ValueError(
            f"Unknown signal_type: {signal_type}. "
            f"Choose from: {SIGNAL_CLASSES}"
        )


def main():
    """CLI entry point for single-signal prediction."""
    parser = argparse.ArgumentParser(
        description="Radar Signal Classifier — Single Prediction"
    )
    parser.add_argument(
        "--model",
        choices=["rf", "cnn"],
        default="rf",
        help="Model to use: 'rf' (Random Forest) or 'cnn' (CNN). Default: rf",
    )
    parser.add_argument(
        "--type",
        choices=SIGNAL_CLASSES,
        default=None,
        help="Signal type to generate and classify",
    )
    parser.add_argument(
        "--freq", type=float, default=5.0, help="Signal frequency in Hz (default: 5.0)"
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=1.0,
        help="Signal amplitude (default: 1.0)",
    )
    parser.add_argument(
        "--file", type=str, default=None, help="Path to a CSV file with signal data"
    )

    args = parser.parse_args()
    predictor = RadarPredictor(model_type=args.model)

    if args.file:
        # Load signal from file
        import pandas as pd

        data = pd.read_csv(args.file)
        signal = data.iloc[:, 0].values
        source = f"file: {args.file}"
    elif args.type:
        # Generate signal
        signal = generate_signal(args.type, freq=args.freq, amplitude=args.amplitude)
        source = f"generated {args.type} (freq={args.freq} Hz, amp={args.amplitude})"
    else:
        # Interactive demo: classify all 4 types
        print("=" * 60)
        print("  Radar Signal Classifier - Demo")
        print(f"    Model: {args.model.upper()}")
        print("=" * 60)

        for sig_type in SIGNAL_CLASSES:
            signal = generate_signal(sig_type)
            result = predictor.predict(signal)

            print(f"\n  Signal: {sig_type.upper()}")
            print(f"  -> Predicted: {result['label'].upper()}")
            print(f"  -> Confidence: {result['confidence']:.1f}%")
            match = "[MATCH]" if result["label"] == sig_type else "[MISMATCH]"
            print(f"  -> Status: {match}")

        print("\n" + "=" * 60)
        return

    # Single prediction
    result = predictor.predict(signal)

    print("=" * 60)
    print("  Radar Signal Classifier")
    print(f"    Source: {source}")
    print(f"    Model: {result['model_type']}")
    print("=" * 60)
    print(f"\n  Predicted Class: {result['label'].upper()}")
    print(f"  Confidence:      {result['confidence']:.1f}%")
    print("\n  Class Probabilities:")
    for cls, prob in sorted(
        result["probabilities"].items(), key=lambda x: x[1], reverse=True
    ):
        bar = "#" * int(prob / 2)
        print(f"    {cls:>8}: {prob:5.1f}% {bar}")

    if "features" in result:
        print(f"\n  Extracted Features:")
        for name, value in result["features"].items():
            print(f"    {name:>20}: {value:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
