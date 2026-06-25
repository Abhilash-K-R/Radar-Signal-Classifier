import os
import sys
import csv

sys.path.append(os.path.dirname(__file__))
from signal_generator import (
    generate_sine_wave,
    generate_square_wave,
    generate_chirp_signal,
    generate_noisy_signal,
    SAMPLE_RATE,
)
from feature_extractor import extract_features

import numpy as np


def build_dataset(n_per_class=1000, output_path="../data/dataset.csv"):
    """
    Generates n_per_class signals for each of the 4 classes,
    extracts features from each, and writes everything to a CSV.
    """
    rows = []

    for i in range(n_per_class):
        # --- Class 1: Sine ---
        freq = np.random.uniform(3, 8)
        amp = np.random.uniform(0.8, 1.2)
        sine = generate_sine_wave(freq, amp)
        sine_feats = extract_features(sine, SAMPLE_RATE)
        sine_feats["label"] = "sine"
        rows.append(sine_feats)

        # --- Class 2: Square ---
        freq_sq = np.random.uniform(3, 8)
        square = generate_square_wave(freq_sq, amp)
        square_feats = extract_features(square, SAMPLE_RATE)
        square_feats["label"] = "square"
        rows.append(square_feats)

        # --- Class 3: Chirp ---
        f0 = np.random.uniform(1, 5)
        f1 = np.random.uniform(30, 60)
        chirp_sig = generate_chirp_signal(f0, f1, amp)
        chirp_feats = extract_features(chirp_sig, SAMPLE_RATE)
        chirp_feats["label"] = "chirp"
        rows.append(chirp_feats)

        # --- Class 4: Noisy ---
        noisy = generate_noisy_signal(freq, amp)
        noisy_feats = extract_features(noisy, SAMPLE_RATE)
        noisy_feats["label"] = "noisy"
        rows.append(noisy_feats)

        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1}/{n_per_class} per class...")

    # Write to CSV
    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDataset built successfully: {len(rows)} total rows")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    build_dataset(n_per_class=1000)