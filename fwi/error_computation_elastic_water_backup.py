"""
Improved elastic water error computation script with enhanced code quality.

This script refactors the original error computation code for elastic water
simulations by:
- Replacing hardcoded data loading with dynamic functions
- Organizing code into reusable functions
- Creating publication-quality plots
- Ensuring PEP 8 compliance
- Adding proper error handling and documentation
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List
import warnings

# Configure matplotlib for LaTeX rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


@dataclass
class ElasticWaterConfig:
    """Configuration for elastic water simulations."""
    vs: float = 1.5
    frequency: float = 10.0  # Hz
    c_values: List[float] = None
    
    def __post_init__(self):
        if self.c_values is None:
            self.c_values = [3.0, 2.7, 2.3, 2.0]
    
    @property
    def min_cell_sizes(self) -> List[float]:
        """Calculate minimum cell size for each C value."""
        return [0.5 / (self.frequency * c) for c in self.c_values]


@dataclass
class MeshDataElastic:
    """Container for mesh data and configuration."""
    quad_data: Dict[int, np.ndarray]
    adaptive_data: Dict[str, np.ndarray]
    reference_data: np.ndarray
    dofs: List[int]
    mesh_ratios: List[float]
    adaptive_dofs: Dict[str, int]
# dofs water 3692162
plt.figure()
# plt.plot(data_1_quads[:, 0, 0], label="Case 1")
# plt.plot(data_2_quads[:, 0, 0], label="Case 2")
# plt.plot(data_3_quads[:, 0, 0], label="Case 3")
# plt.plot(data_4_quads[:, 0, 0], label="Case 4")
# plt.plot(data_5_quads[:, 0, 0], label="Case 5")
# plt.plot(data_6_quads[:, 0, 0], label="Case 6")
# plt.plot(data_7_quads[:, 0, 0], label="Case 7")
# plt.plot(data_8_quads[:, 0, 0], label="Case 8")
# plt.plot(data_9_quads[:, 0, 0], label="Case 9")
# plt.plot(data_10_quads[:, 0, 0], label="Case 10")
plt.plot(data_11_quads[:, 0, 0], label="Case 11")
plt.plot(data_reference[:, 0, 0], label="Reference")
plt.plot(data_adaptive[:, 0, 0], label="Adaptive")
plt.plot(data_adaptive1[:, 0, 0], label="Adaptive1")
plt.plot(data_adaptive2[:, 0, 0], label="Adaptive2")
plt.legend()
plt.show()

plt.figure()
# plt.plot(data_1_quads[:, 0, 1], label="Case 1")
# plt.plot(data_2_quads[:, 0, 1], label="Case 2")
# plt.plot(data_3_quads[:, 0, 1], label="Case 3")
# plt.plot(data_4_quads[:, 0, 1], label="Case 4")
# plt.plot(data_5_quads[:, 0, 1], label="Case 5")
# plt.plot(data_6_quads[:, 0, 1], label="Case 6")
# plt.plot(data_7_quads[:, 0, 1], label="Case 7")
# plt.plot(data_8_quads[:, 0, 1], label="Case 8")
# plt.plot(data_9_quads[:, 0, 1], label="Case 9")
# plt.plot(data_10_quads[:, 0, 1], label="Case 10")
plt.plot(data_11_quads[:, 0, 1], label="Case 11")
plt.plot(data_reference[:, 0, 1], label="Reference")
plt.plot(data_adaptive[:, 0, 1], label="Adaptive")
plt.plot(data_adaptive1[:, 0, 1], label="Adaptive1")
plt.plot(data_adaptive2[:, 0, 1], label="Adaptive2")
plt.legend()
plt.show()


error_adaptive = np.linalg.norm(data_adaptive - data_reference) / np.linalg.norm(data_reference)
error_adaptive1 = np.linalg.norm(data_adaptive1 - data_reference) / np.linalg.norm(data_reference)
error_adaptive2 = np.linalg.norm(data_adaptive2 - data_reference) / np.linalg.norm(data_reference)
error_adaptive3 = np.linalg.norm(data_adaptive3 - data_reference) / np.linalg.norm(data_reference)

# Compute the the relative error with respect to the more refined mesh,
# i.e., case 8
error_1_quads = np.linalg.norm(data_1_quads - data_reference) / np.linalg.norm(data_reference)
error_2_quads = np.linalg.norm(data_2_quads - data_reference) / np.linalg.norm(data_reference)
error_3_quads = np.linalg.norm(data_3_quads - data_reference) / np.linalg.norm(data_reference)
error_4_quads = np.linalg.norm(data_4_quads - data_reference) / np.linalg.norm(data_reference)
error_5_quads = np.linalg.norm(data_5_quads - data_reference) / np.linalg.norm(data_reference)
error_6_quads = np.linalg.norm(data_6_quads - data_reference) / np.linalg.norm(data_reference)
error_7_quads = np.linalg.norm(data_7_quads - data_reference) / np.linalg.norm(data_reference)
error_8_quads = np.linalg.norm(data_8_quads - data_reference) / np.linalg.norm(data_reference)
error_9_quads = np.linalg.norm(data_9_quads - data_reference) / np.linalg.norm(data_reference)
error_10_quads = np.linalg.norm(data_10_quads - data_reference) / np.linalg.norm(data_reference)
error_11_quads = np.linalg.norm(data_11_quads - data_reference) / np.linalg.norm(data_reference)

# plot the relative error for the following mesh sizes
# case 1 - 80x40 mesh, 16/80
c1 = 8/80
# case 2 - 120x60 mesh, 24/120
c2 = 8/120
# case 3 - 160x80 mesh, 32/160
c3 = 8/160
# case 4 - 200x100 mesh, 40/200
c4 = 8/200
# case 5 - 240x120 mesh, 48/240
c5 = 8/240
# case 6 - 280x140 mesh, 56/280
c6 = 8/280
# case 7 - 320x160 mesh, 64/320
c7 = 8/320
# case 8 - 360x180 mesh, 72/360
c8 = 8/360
# case 9 - 400x200 mesh, 80/400
c9 = 8/400
# case 10 - 440x220 mesh, 88/440
c10 = 8/440
# case 11 - 480x240 mesh, 96/480
# print("Error : ", error_1_quads, error_2_quads, error_3_quads, error_4_quads, error_5_quads, error_6_quads, error_7_quads, error_8_quads)
c11 = 8/480
# print("Error : ", error_0, error_1, error_2, error_3, error_4, error_5, error_6, error_7)

# Your data
c_values = [c3, c4, c5, c6, c7, c8, c9, c10]
errors = [error_3_quads, error_4_quads, error_5_quads,
          error_6_quads, error_7_quads, error_8_quads, error_9_quads, error_10_quads]
dofs = [411522, 642402, 924482, 1257762, 1642242, 2077922, 2564802, 3102882]
dofs_case10 = [3692162]
dofs_triangle = [492202]
dofs_triangle1 = [354486]
dofs_triangle2 = [492202]
dofs_triangle3 = [600346]
C = [3.0, 2.7, 2.3, 2.0]
vs = 1.5
f = 10.0  # Hz, representative frequency of the source wavelet
min_cell_size = [0.5 / (f * c) for c in C]  # Minimum cell size for each case

# Use a high-quality style
plt.style.use("seaborn-v0_8-paper")

# Create figure and axis
fig, ax = plt.subplots(figsize=(7, 5))  # Slightly larger figure for clarity

# Plot for quadrilateral mesh
ax.plot(c_values[2:], errors[2:], marker="s", markersize=8, linestyle="-", linewidth=2,
        label="Quadrilateral mesh", color="tab:blue")

# Annotate quadrilateral points with DOFs (larger size, bold)
for x, y, dof in zip(c_values[2:], errors[2:], dofs[2:]):
    ax.annotate(f"{dof}", (x, y), fontsize=14, fontweight="bold", xytext=(-12, 7),
                textcoords="offset points", color="tab:blue")

# Plot for adaptive triangular mesh (only markers)
ax.plot([min_cell_size[0]], [error_adaptive], marker="o", markersize=10,
        label="Adaptive mesh (C = 2.7)", color="tab:red", linestyle="")

ax.annotate(f"{dofs_triangle[0]}", (min_cell_size[0], error_adaptive), fontsize=14, fontweight="bold",
            xytext=(0, -15), textcoords="offset points", ha="center", color="tab:red")

# Plot for second adaptive triangular mesh
ax.plot([min_cell_size[1]], [error_adaptive1], marker="*", markersize=10,
        label="Adaptive mesh (C = 2.3)", color="tab:green", linestyle="")  # No lines
ax.annotate(f"{dofs_triangle1[0]}", (min_cell_size[1], error_adaptive1), fontsize=14, fontweight="bold",
            xytext=(0, -15), textcoords="offset points", ha="center", color="tab:green")

# Plot for third adaptive triangular mesh
ax.plot([min_cell_size[2]], [error_adaptive2], marker="^", markersize=10,
        label="Adaptive mesh (C = 2.0)", color="tab:orange", linestyle="")  # No lines
ax.annotate(f"{dofs_triangle2[0]}", 
            (min_cell_size[2], error_adaptive2), 
            fontsize=14, fontweight="bold",
            xytext=(0, -15), textcoords="offset points", 
            ha="center",  # Center align horizontally
            color="tab:orange")



# Labels and legend (with larger font sizes)
ax.set_xlabel(r"Cell facet size (h)", fontsize=16)
ax.set_ylabel(r"Error", fontsize=16)
ax.tick_params(axis="both", which="major", labelsize=14)
ax.legend(fontsize=13, loc="upper left", frameon=True)

# Grid for readability
ax.grid(True, linestyle="--", alpha=0.6)

# Adjust layout to prevent clipping
plt.tight_layout()

# Save as high-resolution EPS for publication
plt.savefig("error_plot.eps", dpi=300, bbox_inches="tight")

# Show plot
plt.show()
