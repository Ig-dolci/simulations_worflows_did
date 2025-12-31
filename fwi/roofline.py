import matplotlib.pyplot as plt
import numpy as np

# Hardware specifications
memory_bandwidth = 31.2  # GB/s
peak_performance = 51.4  # GFLOP/s
threshold = peak_performance / memory_bandwidth  # FLOP/byte

# Arithmetic intensity range
ai = np.logspace(-1, 2, 100)  # FLOP/byte

# Performance calculation
performance = np.minimum(peak_performance, ai * memory_bandwidth)

# Measured point
measured_ai = 0.0875  # FLOP/byte
measured_gflops = 0.0162  # GFLOPS

# Plot
plt.figure(figsize=(8, 6))
plt.loglog(ai, performance, label="Roofline")
plt.axhline(peak_performance, color='r', linestyle='--', label="Peak Performance")
plt.axvline(threshold, color='g', linestyle='--', label="Threshold")
plt.scatter(measured_ai, measured_gflops, color='b', label="Measured Performance")
plt.xlabel("Arithmetic Intensity (FLOP/byte)")
plt.ylabel("Performance (GFLOP/s)")
plt.title("Roofline Plot")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()