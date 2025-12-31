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


def load_elastic_water_data() -> MeshDataElastic:
    """Load all simulation data from numpy files."""
    print("Loading elastic water simulation data...")
    
    # Load quadrilateral mesh data
    quad_data = {}
    for i in range(1, 12):
        file_path = f"comp_performance/rec_data_case{i}_elastic_water.npy"
        try:
            quad_data[i] = np.load(file_path)
            print(f"Loaded case {i}")
        except FileNotFoundError:
            print(f"Warning: Could not load {file_path}")
    
    # Load reference data
    reference_data = np.load("comp_performance/rec_data_reference_water.npy")
    
    # Load adaptive mesh data
    adaptive_data = {}
    adaptive_files = {
        'adaptive': 'comp_performance/rec_data_adapt_elastic_water.npy',
        'adaptive1': 'comp_performance/rec_data_adapt_elastic_water1.npy',
        'adaptive2': 'comp_performance/rec_data_adapt_elastic_water2.npy',
        'adaptive3': 'comp_performance/rec_data_adapt_elastic_water3.npy'
    }
    
    for key, file_path in adaptive_files.items():
        try:
            adaptive_data[key] = np.load(file_path)
            print(f"Loaded {key}")
        except FileNotFoundError:
            print(f"Warning: Could not load {file_path}")
    
    # Mesh configurations
    dofs = [411522, 642402, 924482, 1257762, 1642242, 2077922, 2564802,
            3102882]
    
    mesh_ratios = [8/160, 8/200, 8/240, 8/280, 8/320, 8/360, 8/400, 8/440]
    
    adaptive_dofs = {
        'adaptive': 492202,
        'adaptive1': 354486,
        'adaptive2': 492202,
        'adaptive3': 600346
    }
    
    return MeshDataElastic(
        quad_data=quad_data,
        adaptive_data=adaptive_data,
        reference_data=reference_data,
        dofs=dofs,
        mesh_ratios=mesh_ratios,
        adaptive_dofs=adaptive_dofs
    )


def compute_elastic_water_errors(mesh_data: MeshDataElastic) -> Dict:
    """Compute relative errors for all mesh configurations."""
    print("Computing relative errors...")
    
    errors = {}
    
    # Compute errors for quadrilateral meshes
    for case_num, data in mesh_data.quad_data.items():
        error = (np.linalg.norm(data - mesh_data.reference_data) /
                np.linalg.norm(mesh_data.reference_data))
        errors[f'case_{case_num}'] = error
        print(f"Case {case_num}: Error = {error:.6f}")
    
    # Compute errors for adaptive meshes
    for key, data in mesh_data.adaptive_data.items():
        error = (np.linalg.norm(data - mesh_data.reference_data) /
                np.linalg.norm(mesh_data.reference_data))
        errors[key] = error
        print(f"{key}: Error = {error:.6f}")
    
    return errors


def plot_data_preview_elastic(mesh_data: MeshDataElastic):
    """Create preview plots of the data."""
    print("Creating data preview plots...")
    
    # Plot first component
    plt.figure(figsize=(10, 6))
    
    # Plot only case 11 and adaptive data to reduce clutter
    if 11 in mesh_data.quad_data:
        plt.plot(mesh_data.quad_data[11][:, 0, 0], label="Case 11")
    
    plt.plot(mesh_data.reference_data[:, 0, 0], label="Reference")
    
    for key, data in mesh_data.adaptive_data.items():
        plt.plot(data[:, 0, 0], label=key.capitalize())
    
    plt.legend()
    plt.title("First Component Comparison")
    plt.tight_layout()
    plt.show()
    
    # Plot second component
    plt.figure(figsize=(10, 6))
    
    if 11 in mesh_data.quad_data:
        plt.plot(mesh_data.quad_data[11][:, 0, 1], label="Case 11")
    
    plt.plot(mesh_data.reference_data[:, 0, 1], label="Reference")
    
    for key, data in mesh_data.adaptive_data.items():
        plt.plot(data[:, 0, 1], label=key.capitalize())
    
    plt.legend()
    plt.title("Second Component Comparison")
    plt.tight_layout()
    plt.show()


def create_elastic_water_error_plot(mesh_data: MeshDataElastic,
                                   errors: Dict,
                                   config: ElasticWaterConfig):
    """Create publication-quality error plot."""
    print("Creating publication-quality error plot...")
    
    # Select subset of quadrilateral data for cleaner plot (cases 3-10)
    selected_cases = list(range(3, 11))
    selected_ratios = mesh_data.mesh_ratios
    selected_errors = [errors[f'case_{i}'] for i in selected_cases]
    selected_dofs = mesh_data.dofs
    
    # Use high-quality style
    plt.style.use("seaborn-v0_8-paper")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Plot quadrilateral mesh (cases 5-10 for cleaner visualization)
    plot_cases = selected_cases[2:]  # Start from case 5
    plot_ratios = selected_ratios[2:]
    plot_errors = selected_errors[2:]
    plot_dofs = selected_dofs[2:]
    
    ax.plot(plot_ratios, plot_errors, marker="s", markersize=8,
            linestyle="-", linewidth=2, label="Quadrilateral mesh",
            color="tab:blue")
    
    # Annotate DOFs
    for x, y, dof in zip(plot_ratios, plot_errors, plot_dofs):
        ax.annotate(f"{dof}", (x, y), fontsize=14, fontweight="bold",
                   xytext=(-12, 7), textcoords="offset points",
                   color="tab:blue")
    
    # Plot adaptive meshes
    adaptive_configs = [
        ('adaptive', 0, "Adaptive mesh (C = 2.7)", "o", "tab:red"),
        ('adaptive1', 1, "Adaptive mesh (C = 2.3)", "*", "tab:green"),
        ('adaptive2', 2, "Adaptive mesh (C = 2.0)", "^", "tab:orange")
    ]
    
    for key, idx, label, marker, color in adaptive_configs:
        if key in errors:
            ax.plot([config.min_cell_sizes[idx]], [errors[key]],
                   marker=marker, markersize=10, label=label,
                   color=color, linestyle="")
            
            # Annotate DOF
            ax.annotate(f"{mesh_data.adaptive_dofs.get(key, 'N/A')}",
                       (config.min_cell_sizes[idx], errors[key]),
                       fontsize=14, fontweight="bold", xytext=(0, -15),
                       textcoords="offset points", ha="center",
                       color=color)
    
    # Formatting
    ax.set_xlabel(r"Cell facet size (h)", fontsize=16)
    ax.set_ylabel(r"Error", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=13, loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    
    # Check if log scale would be appropriate
    error_range = max(plot_errors + [errors.get(key, 0) for key in
                     ['adaptive', 'adaptive1', 'adaptive2']]) / \
                  min(plot_errors + [errors.get(key, 1) for key in
                     ['adaptive', 'adaptive1', 'adaptive2']])
    
    if error_range > 100:
        ax.set_yscale('log')
        print("Using logarithmic scale for y-axis")
    
    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig("elastic_water_error_plot.eps", dpi=300,
                bbox_inches="tight")
    plt.savefig("elastic_water_error_plot.png", dpi=300,
                bbox_inches="tight")
    
    print("Plots saved: elastic_water_error_plot.eps/.png")
    plt.show()


def main():
    """Main execution function."""
    print("Starting elastic water error computation analysis...")
    
    # Load configuration
    config = ElasticWaterConfig()
    
    # Load data
    mesh_data = load_elastic_water_data()
    
    # Compute errors
    errors = compute_elastic_water_errors(mesh_data)
    
    # Create preview plots
    plot_data_preview_elastic(mesh_data)
    
    # Create publication-quality error plot
    create_elastic_water_error_plot(mesh_data, errors, config)
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()
