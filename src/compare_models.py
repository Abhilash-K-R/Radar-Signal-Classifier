"""
Model Comparison: Classical ML vs Deep Learning

Loads pre-trained models (RF, SVM, KNN, CNN) and creates comparison visualizations.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import joblib
import torch

sys.path.append(os.path.dirname(__file__))
from cnn_model import SignalCNN, generate_raw_signals_dataset
from config import RANDOM_STATE
from sklearn.model_selection import train_test_split


def load_classical_models():
    """Load pre-trained classical models."""
    best_model = joblib.load("../results/best_model.pkl")
    return best_model


def load_cnn_model():
    """Load pre-trained CNN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignalCNN(num_classes=4).to(device)
    model.load_state_dict(torch.load("../results/cnn_model.pth", map_location=device))
    model.eval()
    return model, device


def plot_model_comparison():
    """Create a bar chart comparing all models."""
    
    # Model results (from training outputs)
    models = ["Random Forest", "SVM", "KNN", "CNN"]
    accuracies = [0.9563, 0.9463, 0.9413, 1.0000]
    precisions = [0.9565, 0.9489, 0.9421, 1.0000]
    recalls = [0.9563, 0.9463, 0.9413, 1.0000]
    f1_scores = [0.9562, 0.9461, 0.9412, 1.0000]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C5B0D5"]
    
    # Accuracy
    axes[0, 0].bar(models, accuracies, color=colors, alpha=0.8)
    axes[0, 0].set_ylim(0.90, 1.01)
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_title("Accuracy Comparison")
    axes[0, 0].axhline(y=0.95, color="red", linestyle="--", alpha=0.5, label="85% target")
    for i, (model, acc) in enumerate(zip(models, accuracies)):
        axes[0, 0].text(i, acc + 0.003, f"{acc:.3f}", ha="center", fontweight="bold", fontsize=9)
    
    # Precision
    axes[0, 1].bar(models, precisions, color=colors, alpha=0.8)
    axes[0, 1].set_ylim(0.90, 1.01)
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_title("Precision Comparison")
    for i, (model, prec) in enumerate(zip(models, precisions)):
        axes[0, 1].text(i, prec + 0.003, f"{prec:.3f}", ha="center", fontweight="bold", fontsize=9)
    
    # Recall
    axes[1, 0].bar(models, recalls, color=colors, alpha=0.8)
    axes[1, 0].set_ylim(0.90, 1.01)
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_title("Recall Comparison")
    for i, (model, rec) in enumerate(zip(models, recalls)):
        axes[1, 0].text(i, rec + 0.003, f"{rec:.3f}", ha="center", fontweight="bold", fontsize=9)
    
    # F1 Score
    axes[1, 1].bar(models, f1_scores, color=colors, alpha=0.8)
    axes[1, 1].set_ylim(0.90, 1.01)
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_title("F1 Score Comparison")
    for i, (model, f1) in enumerate(zip(models, f1_scores)):
        axes[1, 1].text(i, f1 + 0.003, f"{f1:.3f}", ha="center", fontweight="bold", fontsize=9)
    
    plt.suptitle("Model Performance Comparison: Classical ML vs Deep Learning", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    output_path = os.path.join("..", "results", "model_comparison.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved comparison chart to {output_path}")


def create_summary_table():
    """Print a summary table of all models."""
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    data = [
        ("Random Forest", 0.9563, 0.9565, 0.9563, 0.9562),
        ("SVM", 0.9463, 0.9489, 0.9463, 0.9461),
        ("KNN", 0.9413, 0.9421, 0.9413, 0.9412),
        ("CNN (Deep Learning)", 1.0000, 1.0000, 1.0000, 1.0000),
    ]
    
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-"*80)
    
    for model, acc, prec, rec, f1 in data:
        print(f"{model:<25} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")
    
    print("="*80)
    print("\nKey Insights:")
    print("- Classical ML (RF/SVM/KNN) achieved 94-96% accuracy using hand-engineered features")
    print("- CNN achieved 100% accuracy by learning features directly from raw signals")
    print("- Convergence across all models suggests the classification task is well-defined")
    print("- CNN's perfect accuracy on synthetic data will be tested with noise in Day 14-15")
    print("="*80 + "\n")


if __name__ == "__main__":
    plot_model_comparison()
    create_summary_table()