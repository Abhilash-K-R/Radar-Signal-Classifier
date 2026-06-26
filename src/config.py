"""
Centralized configuration for the Radar Signal Classifier project.
All modules import constants from here.
"""

import numpy as np

# Signal sampling configuration
SAMPLE_RATE = 500          # Points per second
DURATION = 1.0             # Signal length in seconds
N_SAMPLES = int(SAMPLE_RATE * DURATION)
TIME = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)

# ML configuration
TRAIN_TEST_SPLIT = 0.2     # 80/20 train-test split
RANDOM_STATE = 42          # For reproducibility

# Model paths
BEST_MODEL_PATH = "../results/best_model.pkl"
SCALER_PATH = "../results/scaler.pkl"
DATASET_PATH = "../data/dataset.csv"