"""
Centralized configuration for the Radar Signal Classifier project.
All modules import constants from here.
"""

import os
import numpy as np

# --- Project path resolution ---
# All paths are resolved relative to the project root, not the working directory.
# This makes the project work from any directory and in Docker/cloud deployments.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Signal sampling configuration
SAMPLE_RATE = 500          # Points per second
DURATION = 1.0             # Signal length in seconds
N_SAMPLES = int(SAMPLE_RATE * DURATION)
TIME = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)

# ML configuration
TRAIN_TEST_SPLIT = 0.2     # 80/20 train-test split
RANDOM_STATE = 42          # For reproducibility

# Model & data paths (absolute, resolved from project root)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(RESULTS_DIR, "scaler.pkl")
CNN_MODEL_PATH = os.path.join(RESULTS_DIR, "cnn_model.pth")
CNN_METADATA_PATH = os.path.join(RESULTS_DIR, "cnn_metadata.pkl")
DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")