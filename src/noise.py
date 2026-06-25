import numpy as np


def add_gaussian_noise(signal, noise_level=0.5):
    """
    Adds Gaussian (normal distribution) noise to any signal.
    noise_level controls how strong the interference is.
    """
    noise = np.random.normal(loc=0.0, scale=noise_level, size=len(signal))
    return signal + noise


def add_white_noise(signal, noise_level=0.5):
    """
    Adds uniform white noise — every frequency disturbed roughly equally.
    Different statistical flavor than Gaussian, but same purpose:
    simulate real-world interference.
    """
    noise = np.random.uniform(low=-noise_level, high=noise_level, size=len(signal))
    return signal + noise


def add_noise(signal, noise_type="gaussian", noise_level=0.5):
    """
    Single entry point — pick noise type by string.
    Keeps calling code clean: add_noise(my_signal, "gaussian", 0.3)
    """
    if noise_type == "gaussian":
        return add_gaussian_noise(signal, noise_level)
    elif noise_type == "white":
        return add_white_noise(signal, noise_level)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    import sys

    # Allow importing signal_generator from the same src folder
    sys.path.append(os.path.dirname(__file__))
    from signal_generator import generate_sine_wave, TIME

    clean = generate_sine_wave()
    gaussian_version = add_gaussian_noise(clean, noise_level=0.4)
    white_version = add_white_noise(clean, noise_level=0.4)

    fig, axs = plt.subplots(3, 1, figsize=(8, 6))
    axs[0].plot(TIME, clean); axs[0].set_title("Clean Sine Wave")
    axs[1].plot(TIME, gaussian_version); axs[1].set_title("Sine + Gaussian Noise")
    axs[2].plot(TIME, white_version); axs[2].set_title("Sine + White Noise")

    plt.tight_layout()
    output_path = os.path.join("..", "results", "noise_examples.png")
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    print("Noise injection working correctly!")