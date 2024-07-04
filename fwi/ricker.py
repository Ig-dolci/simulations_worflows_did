import numpy as np
import matplotlib.pyplot as plt

def ricker_wavelet(t, f):
    """
    Compute the Ricker wavelet (Mexican hat wavelet) for a given time series t and central frequency f.

    Parameters:
    t (numpy.ndarray): Time array.
    f (float): Central frequency.

    Returns:
    numpy.ndarray: Values of the Ricker wavelet at the given times.
    """
    a = 2 * (np.pi * f)**2
    return (1 - a * t**2) * np.exp(-a * t**2 / 2)

# Define time vector centered around zero
t = np.linspace(-1, 1, 1000)  # Time vector

# Define central frequency of the wavelet
f = 5  # Central frequency in Hz

# Generate Ricker wavelet
wavelet = ricker_wavelet(t, f)

# Plot the Ricker wavelet
plt.figure(figsize=(10, 6))
plt.plot(t, wavelet)
plt.title(f'Ricker Wavelet (Central frequency: {f} Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
