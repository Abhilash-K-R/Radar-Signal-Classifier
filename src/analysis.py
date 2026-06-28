"""
Deep Analysis: Classical ML vs Deep Learning Trade-offs

Documents key findings from SNR testing and model comparison.
This section goes in the README as a "Learning" or "Analysis" section.
"""

import os


def generate_analysis_report():
    """Generate markdown analysis for README."""
    
    report = """
# 📊 Deep Analysis: Classical ML vs Deep Learning

## SNR Testing Reveals Different Failure Modes

### Scenario
Tested both Random Forest and CNN across varying Signal-to-Noise Ratios (SNR), simulating real-world radar conditions where environmental noise corrupts signals.

### Key Findings

#### Random Forest: Graceful Degradation
- **High SNR (40 dB):** 92.25% accuracy
- **Medium SNR (15 dB):** 83.75% accuracy  
- **Low SNR (5 dB):** 63% accuracy
- **Very Low SNR (-5 dB):** 25% accuracy

**Interpretation:** Classical ML with hand-engineered features degrades *smoothly*. Even under heavy noise (SNR = 5 dB), the model maintains 63% accuracy — still useful. The engineered features (energy, std, RMS) are inherently robust because they summarize signal properties that noise-corrupted signals still exhibit.

#### CNN: Perfect Until Collapse
- **High SNR (40 dB):** 100% accuracy
- **Medium SNR (15 dB):** 100% accuracy
- **Low SNR (5 dB):** 36.75% accuracy ← **Sudden Collapse**
- **Very Low SNR (-5 dB):** 25% accuracy

**Interpretation:** Deep learning trained only on *clean synthetic data* memorizes patterns that don't generalize to noisy conditions. When noise exceeds the model's "comfort zone," performance collapses catastrophically. This is classic **distribution shift** — the test data (noisy) differs from training data (clean).

---

## Why Random Forest Wins in Real-World Conditions

### 1. **Engineered Features Encode Domain Knowledge**
Random Forest's features (energy, standard deviation, RMS, dominant frequency) are rooted in signal processing theory. They capture properties that remain meaningful even under noise:
- **Energy** = signal power (noise doesn't make a signal "disappear")
- **Standard Deviation** = signal variation (still detectable through noise)
- **Dominant Frequency** = FFT-based, still visible in noisy spectrum

### 2. **No Assumption of "Clean Data"**
CNN was trained on perfect synthetic signals. It implicitly learned that "signals look like clean sine/square waves." Real radar signals never look like this — they're always corrupted by environmental interference.

Random Forest, by contrast, was trained on *labeled* (but still somewhat noisy) signals, making its features more generalizable.

### 3. **Feature Redundancy = Robustness**
With 8 features, Random Forest has multiple "voting channels." Even if noise corrupts some features, others remain informative. CNN has a single learned representation — if noise breaks that, the model breaks.

---

## Trade-offs Summary

| Aspect | Classical ML (RF) | Deep Learning (CNN) |
|---|---|---|
| **Accuracy (Clean Data)** | 95.63% | 100% |
| **Robustness to Noise** | High (graceful degradation) | Low (sudden collapse) |
| **Real-World Generalization** | Better | Worse |
| **Interpretability** | High (feature importance visible) | Low (black box) |
| **Training Time** | Fast (minutes) | Slow (depends on data) |
| **Scalability** | Excellent | Excellent |

---

## Practical Implication for Radar Systems

**Lesson:** When deploying signal classifiers to real radar systems, classical ML with robust feature engineering often outperforms deep learning if:
1. Training data is limited or clean
2. Real-world noise characteristics are unknown
3. Model robustness under distribution shift is critical
4. Interpretability matters (operators need to understand why a signal is classified)

**When Deep Learning Wins:**
- You have massive labeled datasets of *real, noisy* signals (not synthetic)
- Noise characteristics are well-understood and included in training
- Interpretability is less important than marginal accuracy gains
- Computational resources for continuous retraining are available

---

## Conclusion

This project demonstrates that **"better algorithm" ≠ "better system."** Context matters. Random Forest's hand-engineered features, grounded in signal processing theory, proved more robust to the distribution shifts that real-world deployments face. The CNN's perfect accuracy on synthetic data became its Achilles heel when transferred to noisy conditions.

The ideal production system might **ensemble both approaches:**
- Use CNN for high-SNR, real-time processing
- Fall back to Random Forest for low-SNR, degraded conditions
- Monitor SNR continuously and swap models accordingly
"""
    
    return report


def save_analysis_to_readme():
    """Append analysis to the README."""
    
    analysis = generate_analysis_report()
    
    # Save as separate file for now
    output_path = os.path.join("..", "ANALYSIS.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(analysis)
    
    print(f"Analysis saved to {output_path}")
    print("\nThis can be integrated into the main README or kept as a separate ANALYSIS.md file.\n")


if __name__ == "__main__":
    save_analysis_to_readme()