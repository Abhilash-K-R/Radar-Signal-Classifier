import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from scipy.fft import fft, fftfreq

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from signal_generator import (
    generate_sine_wave,
    generate_square_wave,
    generate_chirp_signal,
    generate_noisy_signal,
    SAMPLE_RATE,
    TIME,
)
from feature_extractor import extract_features
from noise import add_noise


# Page config
st.set_page_config(page_title="Radar Signal Classifier", layout="wide")

st.title("🛰️ Radar Signal Classification Simulator")
st.markdown("""
An interactive ML-powered classifier that distinguishes between different radar signal types.
Generate or upload a signal, extract features, and see live predictions.
""")

# Load the trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'best_model.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'scaler.pkl')
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model_and_scaler()

# Sidebar for input method
st.sidebar.header("⚙️ Input Configuration")
input_method = st.sidebar.radio("Choose input method:", ["Generate Signal", "Upload CSV"])

signal = None
signal_name = ""

# --- Method 1: Generate a signal ---
if input_method == "Generate Signal":
    st.sidebar.subheader("Signal Parameters")
    signal_type = st.sidebar.selectbox(
        "Select signal type:",
        ["sine", "square", "chirp", "noisy"]
    )

    if signal_type == "sine":
        freq = st.sidebar.slider("Frequency (Hz)", 1.0, 20.0, 5.0)
        amp = st.sidebar.slider("Amplitude", 0.1, 2.0, 1.0)
        signal = generate_sine_wave(freq, amp)
        signal_name = f"Sine (f={freq:.1f} Hz, A={amp:.2f})"

    elif signal_type == "square":
        freq = st.sidebar.slider("Frequency (Hz)", 1.0, 20.0, 5.0)
        amp = st.sidebar.slider("Amplitude", 0.1, 2.0, 1.0)
        signal = generate_square_wave(freq, amp)
        signal_name = f"Square (f={freq:.1f} Hz, A={amp:.2f})"

    elif signal_type == "chirp":
        f0 = st.sidebar.slider("Start Frequency (Hz)", 1.0, 10.0, 2.0)
        f1 = st.sidebar.slider("End Frequency (Hz)", 20.0, 100.0, 50.0)
        amp = st.sidebar.slider("Amplitude", 0.1, 2.0, 1.0)
        signal = generate_chirp_signal(f0, f1, amp)
        signal_name = f"Chirp (f0={f0:.1f}, f1={f1:.1f} Hz, A={amp:.2f})"

    elif signal_type == "noisy":
        freq = st.sidebar.slider("Base Frequency (Hz)", 1.0, 20.0, 5.0)
        amp = st.sidebar.slider("Amplitude", 0.1, 2.0, 1.0)
        noise_level = st.sidebar.slider("Noise Level", 0.1, 3.0, 1.5)
        signal = generate_noisy_signal(freq, amp, noise_level)
        signal_name = f"Noisy (f={freq:.1f} Hz, noise={noise_level:.2f})"

# --- Method 2: Upload a signal ---
else:
    uploaded_file = st.sidebar.file_uploader("Upload a CSV with signal data", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        # Assume first column is the signal
        signal = data.iloc[:, 0].values
        signal_name = uploaded_file.name

# --- Prediction Pipeline ---
if signal is not None:
    st.success(f"✅ Signal loaded: {signal_name}")

    # Extract features
    features = extract_features(signal, SAMPLE_RATE)
    features_df = pd.DataFrame([features])

    # Scale features
    features_scaled = scaler.transform(features_df)

    # Predict
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    confidence = np.max(probabilities) * 100

    # Display prediction prominently
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Prediction Result")
        st.metric(
            label="Predicted Signal Type",
            value=prediction.upper(),
            delta=f"Confidence: {confidence:.1f}%"
        )

    with col2:
        st.subheader("📊 Confidence Scores")
        class_names = model.classes_
        conf_data = pd.DataFrame({
            "Signal Type": class_names,
            "Confidence (%)": probabilities * 100
        }).sort_values("Confidence (%)", ascending=False)
        st.dataframe(conf_data, use_container_width=True)

    # Display extracted features
    st.subheader("📈 Extracted Features")
    features_display = pd.DataFrame({
        "Feature": list(features.keys()),
        "Value": list(features.values())
    })
    st.dataframe(features_display, use_container_width=True)

    # Visualizations
    st.subheader("📉 Signal Visualization")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(TIME, signal, linewidth=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Input Signal (Time Domain)")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with col2:
        # FFT visualization
        n = len(signal)
        fft_vals = np.abs(fft(signal))[:n // 2]
        freqs = fftfreq(n, d=1 / SAMPLE_RATE)[:n // 2]

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(freqs, fft_vals, linewidth=1)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_title("Signal Spectrum (FFT)")
        ax.set_xlim(0, 50)
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    # Footer
    st.divider()
    st.markdown("""
    **About this classifier:**
    - Trained on 4000 synthetic radar signals across 4 classes
    - Uses Random Forest algorithm with 95.6% accuracy
    - Features: mean, std, RMS, energy, peak-to-peak, dominant frequency
    - Model trained on extracted statistical and frequency-domain features
    """)

else:
    st.info("👆 Select an input method in the sidebar to get started!")