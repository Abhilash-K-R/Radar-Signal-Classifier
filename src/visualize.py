import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.append(os.path.dirname(__file__))
from train_model import train_and_evaluate, load_dataset


def plot_confusion_matrices(trained_models, X_test, y_test, output_dir="../results"):
    """One confusion matrix per model, saved as separate images."""
    labels = sorted(y_test.unique())

    for name, model in trained_models.items():
        predictions = model.predict(X_test)
        cm = confusion_matrix(y_test, predictions, labels=labels)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {name}")

        safe_name = name.lower().replace(" ", "_")
        path = os.path.join(output_dir, f"confusion_matrix_{safe_name}.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)
        print(f"Saved {path}")


def plot_accuracy_comparison(results, output_dir="../results"):
    """Bar chart comparing accuracy across models."""
    names = list(results.keys())
    accuracies = [results[name]["accuracy"] for name in names]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, accuracies, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                 f"{acc:.3f}", ha="center", fontweight="bold")

    path = os.path.join(output_dir, "accuracy_comparison.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_feature_importance(rf_model, feature_names, output_dir="../results"):
    """Random Forest's built-in feature importance scores."""
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(range(len(importances)), importances[sorted_idx], color="#4C72B0")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in sorted_idx], rotation=45, ha="right")
    ax.set_title("Feature Importance — Random Forest")
    ax.set_ylabel("Importance")

    path = os.path.join(output_dir, "feature_importance.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    results, trained_models, X_test_scaled, y_test = train_and_evaluate()
    X, y = load_dataset()
    feature_names = list(X.columns)

    plot_confusion_matrices(trained_models, X_test_scaled, y_test)
    plot_accuracy_comparison(results)
    plot_feature_importance(trained_models["Random Forest"], feature_names)

    print("\nAll visualizations saved to results/")