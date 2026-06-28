"""
SNR (Signal-to-Noise Ratio) Testing

Tests both CNN and Random Forest across varying noise levels to show
how accuracy degrades as signals become noisier. This is the real-world test.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import joblib
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(__file__))
from signal_generator import (
    generate_sine_wave,
    generate_square_wave,
    generate_chirp_signal,
    generate_noisy_signal,
    SAMPLE_RATE,
)
from noise import add_noise
from feature_extractor import extract_features
from cnn_model import SignalCNN


def generate_signals_with_snr(n_per_class=100, snr_db=20):
    """
    Generate signals at a specific SNR level.
    
    SNR (dB) = 20 * log10(signal_power / noise_power)
    Higher SNR = cleaner signal, Lower SNR = noisier signal
    """
    signals = []
    labels = []
    
    # Convert SNR from dB to linear ratio
    snr_linear = 10 ** (snr_db / 20)
    
    for i in range(n_per_class):
        freq = np.random.uniform(3, 8)
        amp = np.random.uniform(0.8, 1.2)
        
        # Generate clean signal
        sine = generate_sine_wave(freq, amp)
        # Add noise inversely proportional to SNR
        noise_level = amp / snr_linear
        sine_noisy = add_noise(sine, "gaussian", noise_level)
        signals.append(sine_noisy)
        labels.append(0)
        
        freq_sq = np.random.uniform(3, 8)
        square = generate_square_wave(freq_sq, amp)
        square_noisy = add_noise(square, "gaussian", noise_level)
        signals.append(square_noisy)
        labels.append(1)
        
        f0 = np.random.uniform(1, 5)
        f1 = np.random.uniform(30, 60)
        chirp = generate_chirp_signal(f0, f1, amp)
        chirp_noisy = add_noise(chirp, "gaussian", noise_level)
        signals.append(chirp_noisy)
        labels.append(2)
        
        noisy = generate_noisy_signal(freq, amp)
        noisy_noisier = add_noise(noisy, "gaussian", noise_level)
        signals.append(noisy_noisier)
        labels.append(3)
    
    return np.array(signals), np.array(labels)


def evaluate_rf_at_snr(rf_model, scaler, signals, labels):
    """Evaluate Random Forest at a specific SNR level."""
    features = []
    for signal in signals:
        feat_dict = extract_features(signal, SAMPLE_RATE)
        features.append(list(feat_dict.values()))
    
    features = np.array(features)
    features_scaled = scaler.transform(features)
    predictions = rf_model.predict(features_scaled)
    
    # Map numeric labels to string labels (what the model was trained on)
    label_map = {0: "sine", 1: "square", 2: "chirp", 3: "noisy"}
    labels_str = np.array([label_map[l] for l in labels])
    
    accuracy = accuracy_score(labels_str, predictions)
    
    return accuracy


def evaluate_cnn_at_snr(cnn_model, device, signals, labels):
    """Evaluate CNN at a specific SNR level."""
    # Convert to tensor
    X_tensor = torch.FloatTensor(signals).unsqueeze(1).to(device)
    y_tensor = torch.LongTensor(labels)
    
    cnn_model.eval()
    with torch.no_grad():
        outputs = cnn_model(X_tensor)
        _, predictions = torch.max(outputs, 1)
        accuracy = (predictions.cpu().numpy() == y_tensor.numpy()).mean()
    
    return accuracy


def run_snr_test():
    """Run the full SNR testing suite."""
    
    # Load models
    print("Loading pre-trained models...")
    rf_model = joblib.load("../results/best_model.pkl")
    scaler = joblib.load("../results/scaler.pkl")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = SignalCNN(num_classes=4).to(device)
    cnn_model.load_state_dict(torch.load("../results/cnn_model.pth", map_location=device))
    cnn_model.eval()
    
    # SNR levels to test (dB)
    snr_levels = [40, 30, 20, 15, 10, 5, 0, -5]
    rf_accuracies = []
    cnn_accuracies = []
    
    print(f"\nTesting across SNR levels (higher dB = cleaner signal)...\n")
    print(f"{'SNR (dB)':<12} {'Random Forest':<20} {'CNN':<20}")
    print("-" * 52)
    
    for snr in snr_levels:
        # Generate test signals at this SNR
        signals, labels = generate_signals_with_snr(n_per_class=100, snr_db=snr)
        
        # Evaluate both models
        rf_acc = evaluate_rf_at_snr(rf_model, scaler, signals, labels)
        cnn_acc = evaluate_cnn_at_snr(cnn_model, device, signals, labels)
        
        rf_accuracies.append(rf_acc)
        cnn_accuracies.append(cnn_acc)
        
        print(f"{snr:<12} {rf_acc:<20.4f} {cnn_acc:<20.4f}")
    
    print("-" * 52 + "\n")
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(snr_levels, rf_accuracies, "o-", linewidth=2, markersize=8, label="Random Forest", color="#4C72B0")
    ax.plot(snr_levels, cnn_accuracies, "s-", linewidth=2, markersize=8, label="CNN", color="#C5B0D5")
    
    ax.set_xlabel("SNR (dB) — Higher = Cleaner Signal", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Model Robustness: Accuracy vs Signal-to-Noise Ratio", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add annotation
    ax.text(0.98, 0.05, "Lower SNR = More Realistic Conditions", 
            transform=ax.transAxes, ha="right", fontsize=9, style="italic",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
    
    output_path = os.path.join("..", "results", "snr_degradation.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"SNR testing plot saved to {output_path}")
    
    # Analysis
    print("\nSNR Testing Analysis:")
    print("="*60)
    rf_drop = rf_accuracies[0] - rf_accuracies[-1]
    cnn_drop = cnn_accuracies[0] - cnn_accuracies[-1]
    
    print(f"Random Forest:")
    print(f"  - Accuracy at high SNR (40 dB): {rf_accuracies[0]:.4f}")
    print(f"  - Accuracy at low SNR (-5 dB):  {rf_accuracies[-1]:.4f}")
    print(f"  - Degradation:                  {rf_drop:.4f} ({rf_drop*100:.2f}%)")
    
    print(f"\nCNN:")
    print(f"  - Accuracy at high SNR (40 dB): {cnn_accuracies[0]:.4f}")
    print(f"  - Accuracy at low SNR (-5 dB):  {cnn_accuracies[-1]:.4f}")
    print(f"  - Degradation:                  {cnn_drop:.4f} ({cnn_drop*100:.2f}%)")
    
    if rf_drop < cnn_drop:
        print(f"\n✓ Random Forest is more robust to noise (lower degradation)")
    else:
        print(f"\n✓ CNN is more robust to noise (lower degradation)")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    run_snr_test()