# 🛰️ Radar Signal Classification Simulator

> An educational ML-powered simulator that generates synthetic radar signals, extracts features, and classifies different signal types using machine learning. Built as a portfolio project demonstrating signal processing, feature engineering, and classical ML workflows.

**Status:** ✅ Complete (95.6% accuracy)

---

## 📋 Overview

This project implements a complete machine learning pipeline for radar signal classification:

1. **Signal Generation** — Synthetic radar-like signals (sine, square, chirp, noisy)
2. **Noise Injection** — Gaussian and white noise simulation
3. **Feature Extraction** — Statistical and frequency-domain features via FFT
4. **Model Training** — Random Forest, SVM, KNN classifiers
5. **Evaluation** — Accuracy, precision, recall, F1 score metrics
6. **Visualization** — Confusion matrices, FFT plots, feature importance
7. **Interactive Dashboard** — Streamlit app for real-time predictions

**Note:** This is an educational simulator, not a real military radar system.

---

## 📊 Quick Stats

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-95.6%25-brightgreen)
![Models](https://img.shields.io/badge/Models-3%20(RF%2C%20SVM%2C%20KNN)-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

---

## 🎯 Problem Statement

Radar systems receive multiple signal types that must be classified before further processing. This project addresses the challenge of distinguishing between four signal patterns:
- **Sine Wave** — Stable periodic return
- **Square Wave** — Alternative periodic pattern
- **Chirp Signal** — Frequency sweeping over time (common in real radar)
- **Noisy Signal** — Clean signal buried in environmental interference

The goal is to build a classifier that generalizes to unseen signals and exceeds 85% accuracy.

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Random Forest | **95.63%** | 95.65% | 95.63% | 95.62% |
| SVM | 94.63% | 94.89% | 94.63% | 94.61% |
| KNN | 94.13% | 94.21% | 94.13% | 94.12% |

**Key Finding:** Energy and standard deviation were the most important features (17.8% importance each), indicating that magnitude-based features outweighed frequency-based features for this classification task.

---

## 🛠️ Architecture
Signal Generator (sine, square, chirp, noisy)

↓

Noise Injection (Gaussian, white)

↓

Feature Extractor (8 features: mean, std, max, min, RMS, energy, peak-to-peak, dominant_frequency)

↓

Dataset Builder (4000 labeled samples)

↓

ML Training Engine (Random Forest, SVM, KNN)

↓

Model Evaluation & Visualization

↓

Interactive Streamlit Dashboard
---

## 📁 Project Structure
Radar-Signal-Classifier/

│

├── src/

│   ├── config.py                    # Centralized constants

│   ├── signal_generator.py          # Synthetic signal generation

│   ├── noise.py                     # Gaussian and white noise injection

│   ├── feature_extractor.py         # FFT and statistical features

│   ├── dataset_builder.py           # Generate 4000-row dataset

│   ├── train_model.py               # Train RF/SVM/KNN, save best model

│   ├── predictor.py                 # Single-signal prediction utility

│   └── visualize.py                 # Generate confusion matrices, charts

│

├── dashboard/

│   └── app.py                       # Streamlit interactive dashboard

│

├── data/

│   ├── generated_signals/           # Raw signal storage

│   └── dataset.csv                  # 4000 labeled feature vectors

│

├── results/

│   ├── best_model.pkl               # Trained Random Forest classifier

│   ├── scaler.pkl                   # StandardScaler for feature normalization

│   ├── confusion_matrix_*.png       # Per-model confusion matrices

│   ├── accuracy_comparison.png      # Bar chart of model performance

│   ├── feature_importance.png       # Random Forest feature weights

│   ├── signal_examples.png          # Raw signal visualizations

│   ├── noise_examples.png           # Noise injection demonstration

│   └── fft_check.png                # FFT validation plot

│

├── notebooks/

│   └── experimentation.ipynb        # Jupyter notebook for exploration

│

├── requirements.txt                 # Python dependencies

├── .gitignore                       # Exclude generated files

└── README.md                        # This file
---
## ⚡ Quick Start (TL;DR)

```bash
git clone https://github.com/Abhilash-K-R/Radar-Signal-Classifier.git
cd Radar-Signal-Classifier
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cd src
python dataset_builder.py && python train_model.py && python visualize.py
cd ../dashboard
streamlit run app.py
```

Then visit `http://localhost:8501`

## 🚀 How to Run

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/Abhilash-K-R/Radar-Signal-Classifier.git
cd Radar-Signal-Classifier

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Generate Dataset & Train Model

```bash
cd src

# Generate 4000 synthetic signals with extracted features
python dataset_builder.py

# Train classifiers and save the best model
python train_model.py

# Generate visualizations
python visualize.py
```

### Launch Interactive Dashboard

```bash
cd dashboard
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 📈 Feature Importance

Random Forest identified these features as most predictive (in order):

1. **Energy** (17.8%) — Signal power; noisy signals have dramatically higher energy
2. **Standard Deviation** (17.8%) — Magnitude variation; stable signals differ from noisy
3. **RMS** (16.1%) — Root mean square; captures overall signal strength regardless of sign
4. **Dominant Frequency** (14.0%) — FFT-based peak frequency; distinguishes periodic patterns
5. Others (< 11% each)

**Insight:** Magnitude-based features dominated over frequency-domain features, suggesting that energy and variance alone provide strong separation between signal types.

---

## 🔬 Methodology

### Phase 1: Signal Generation
Generated 1000 samples per class × 4 classes = 4000 total signals:
- **Sine:** `sin(2πft)` at random frequencies (3–8 Hz) and amplitudes (0.8–1.2)
- **Square:** `square(2πft)` with same frequency/amplitude randomization
- **Chirp:** Frequency sweeps from `f0` (1–5 Hz) to `f1` (30–60 Hz) linearly
- **Noisy:** Sine wave + Gaussian noise (σ = 1.5)

### Phase 2: Noise Injection
Applied optional Gaussian and white noise to simulate real-world interference:
- Gaussian: `np.random.normal(μ=0, σ=noise_level)`
- White: `np.random.uniform(-noise_level, +noise_level)`

### Phase 3: Feature Extraction
Extracted 8 features per signal:
- **Time-domain:** mean, std, max, min, RMS, energy, peak-to-peak
- **Frequency-domain:** dominant frequency (via scipy.fft)

### Phase 4: Dataset Creation
Constructed a CSV with 4000 rows and 9 columns (8 features + 1 label).

### Phase 5: Model Training
- **Train/Test Split:** 80/20 with stratification to preserve class balance
- **Scaling:** StandardScaler (SVM and KNN require normalized features)
- **Models Trained:**
  - Random Forest: 100 trees, `random_state=42`
  - SVM: RBF kernel
  - KNN: k=5 neighbors

### Phase 6: Evaluation
Compared using standard ML metrics: accuracy, precision, recall, F1 score.

### Phase 7: Dashboard
Built interactive Streamlit app allowing users to:
- Generate signals with custom parameters
- Upload CSV signal data
- Extract features in real-time
- Receive predictions with confidence scores
- Visualize signals in time and frequency domains

---

## 💡 Key Learnings

1. **Feature Engineering > Algorithm Choice:** All three models converged around 94–96% accuracy, showing that good features matter more than algorithmic sophistication.

2. **Convergence as Validation:** Multiple independent algorithms arriving at similar performance suggests the classification task is genuinely solvable, not a lucky outlier.

3. **Energy Dominates:** Magnitude-based features were ~2× more important than frequency features for this problem, which aligns with domain knowledge (noisy signals have inherently higher power).

4. **Reproducibility Matters:** Using `random_state=42` and version-controlled training makes results auditable and repeatable — critical for scientific credibility.

---

## 🔮 Future Scope

- **Deep Learning:** Train CNN/LSTM models directly on raw signals (no manual feature extraction) and compare against classical ML
- **Real Radar Data:** Integrate with actual radar datasets (once available) to test generalization
- **Incremental Learning:** Implement a pipeline to retrain on new labeled signals without restarting from scratch
- **Model Versioning:** Track multiple model versions, performance metrics, and deployment timestamps
- **Noise Robustness:** Test classifier performance across SNR (signal-to-noise ratio) levels to understand degradation curves
- **Explainability:** Add SHAP or LIME for per-prediction feature attribution
- **Web Deployment:** Host the dashboard on AWS/GCP for broader accessibility

---

## 🛡️ Technologies Used

**Core ML & Data Processing:**
- NumPy — Numerical computing
- Pandas — Data manipulation
- Scikit-learn — Classical ML (Random Forest, SVM, KNN)
- SciPy — Signal processing (FFT, chirp generation)

**Visualization & Dashboard:**
- Matplotlib — Static plots and charts
- Streamlit — Interactive web dashboard

**Development & Deployment:**
- Git/GitHub — Version control
- Joblib — Model serialization

---

## 👤 Author

**Abhilash K R**  
Computer Science Engineering, 2027 batch  
Shridevi Institute of Engineering and Technology (SIET), Tumakuru, Karnataka

**GitHub:** [github.com/Abhilash-K-R](https://github.com/Abhilash-K-R)  
**Portfolio:** AKR Studio (AI-powered development)

---
---

## 📸 Screenshots

**Dashboard Prediction:**
- Interactive signal generation with live classification
- Confidence scores for each signal type
- Real-time FFT visualization

**Evaluation Metrics:**
- Confusion matrices for all 3 models
- Accuracy comparison bar chart
- Feature importance rankings

![Feature Importance](results/feature_importance.png)
![Accuracy Comparison](results/accuracy_comparison.png)

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- Inspired by real radar signal processing and domain requirements
- Built as part of structured portfolio development for internship applications
- Developed with focus on clean code, reproducibility, and professional documentation