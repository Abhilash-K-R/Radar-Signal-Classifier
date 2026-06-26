import numpy as np
from config import SAMPLE_RATE, DURATION, N_SAMPLES, TIME


def generate_sine_wave(freq=5.0, amplitude=1.0):
    """Class 1: Sine wave — represents a stable radar return."""
    signal = amplitude * np.sin(2 * np.pi * freq * TIME)
    return signal


def generate_square_wave(freq=5.0, amplitude=1.0):
    """Class 2: Square wave — a different stable signal pattern."""
    from scipy import signal as sp_signal
    signal = amplitude * sp_signal.square(2 * np.pi * freq * TIME)
    return signal


def generate_chirp_signal(f0=2.0, f1=50.0, amplitude=1.0):
    """Class 3: Chirp — frequency increases over time. Common in real radar."""
    from scipy.signal import chirp
    signal = amplitude * chirp(TIME, f0=f0, f1=f1, t1=DURATION, method='linear')
    return signal


def generate_noisy_signal(freq=5.0, amplitude=1.0, noise_level=1.5):
    """Class 4: A sine wave buried under heavy noise."""
    base = generate_sine_wave(freq, amplitude)
    heavy_noise = np.random.normal(0, noise_level, N_SAMPLES)
    return base + heavy_noise


def generate_dataset_raw(n_per_class=1000):
    """
    Generates n_per_class raw signals for each of the 4 classes.
    Returns a dict: {label: list_of_signals}
    """
    dataset = {
        "sine": [],
        "square": [],
        "chirp": [],
        "noisy": []
    }

    for _ in range(n_per_class):
        freq = np.random.uniform(3, 8)
        amp = np.random.uniform(0.8, 1.2)
        dataset["sine"].append(generate_sine_wave(freq, amp))

        freq_sq = np.random.uniform(3, 8)
        dataset["square"].append(generate_square_wave(freq_sq, amp))

        f0 = np.random.uniform(1, 5)
        f1 = np.random.uniform(30, 60)
        dataset["chirp"].append(generate_chirp_signal(f0, f1, amp))

        dataset["noisy"].append(generate_noisy_signal(freq, amp))

    return dataset


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os

    sine = generate_sine_wave()
    square = generate_square_wave()
    chirp_sig = generate_chirp_signal()
    noisy = generate_noisy_signal()

    fig, axs = plt.subplots(4, 1, figsize=(8, 8))
    axs[0].plot(TIME, sine); axs[0].set_title("Sine Wave (Class 1)")
    axs[1].plot(TIME, square); axs[1].set_title("Square Wave (Class 2)")
    axs[2].plot(TIME, chirp_sig); axs[2].set_title("Chirp Signal (Class 3)")
    axs[3].plot(TIME, noisy); axs[3].set_title("Noisy Signal (Class 4)")

    plt.tight_layout()

    output_path = os.path.join("..", "results", "signal_examples.png")
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    print("All 4 signal types generated successfully!")