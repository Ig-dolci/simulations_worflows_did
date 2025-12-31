"""
Improved elastic water error computation script with enhanced code quality.

This script computes and visualizes errors for elastic water simulations using:
- Dynamic data loading functions
- Modular, reusable code structure
- Publication-quality plots
- PEP 8 compliance
- Proper error handling and documentation
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple
import os
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


def load_elastic_simulation_data() -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Load all simulation data dynamically.
    
    Returns:
        Tuple containing dictionaries of simulation data and reference data
    """
    data_dict = {}
    
    # Load quadrilateral mesh data (cases 1-11)
    for i in range(1, 12):
        filename = f"data_{i}_quads.npy"
        if os.path.exists(filename):
            data_dict[f'quad_{i}'] = np.load(filename)
            print(f"Loaded {filename}")
        else:
            print(f"Warning: {filename} not found")
    
    # Load adaptive mesh data
    adaptive_files = [
        ('adaptive', 'data_adaptive.npy'),
        ('adaptive1', 'data_adaptive1.npy'),
        ('adaptive2', 'data_adaptive2.npy'),
        ('adaptive3', 'data_adaptive3.npy')
    ]
    
    for key, filename in adaptive_files:
        if os.path.exists(filename):
            data_dict[key] = np.load(filename)
            print(f"Loaded {filename}")
        else:
            print(f"Warning: {filename} not found")
    
    # Load reference data
    reference_filename = "data_reference.npy"
    if os.path.exists(reference_filename):
        reference_data = np.load(reference_filename)
        print(f"Loaded {reference_filename}")
    else:
        raise FileNotFoundError(f"Reference data file {reference_filename} "
                               f"not found")
    
    return data_dict, reference_data


def compute_elastic_errors(data_dict: Dict[str, np.ndarray],
                          reference_data: np.ndarray) -> Tuple[List[float],
                                                               List[float]]:
    """
    Compute relative errors for all simulation cases.
    
    Args:
        data_dict: Dictionary containing all simulation data
        reference_data: Reference solution data
    
    Returns:
        Tuple of (quadrilateral errors, adaptive errors)
    """
    reference_norm = np.linalg.norm(reference_data)
    
    # Compute errors for quadrilateral meshes
    quad_errors = []
    for i in range(1, 12):
        key = f'quad_{i}'
        if key in data_dict:
            error = (np.linalg.norm(data_dict[key] - reference_data) / 
                    reference_norm)
            quad_errors.append(error)
        else:
            quad_errors.append(None)
    
    # Compute errors for adaptive meshes
    adaptive_errors = []
    adaptive_keys = ['adaptive', 'adaptive1', 'adaptive2', 'adaptive3']
    for key in adaptive_keys:
        if key in data_dict:
            error = (np.linalg.norm(data_dict[key] - reference_data) / 
                    reference_norm)
            adaptive_errors.append(error)
        else:
            adaptive_errors.append(None)
    
    return quad_errors, adaptive_errors


def get_elastic_mesh_configurations() -> Tuple[List[float], List[int], 
                                             List[float], List[int]]:
    """
    Get mesh configuration data for elastic water simulations.
    
    Returns:
        Tuple of (c_values for quads, dofs for quads, 
                 min_cell_sizes for adaptive, dofs for adaptive)
    """
    # Quadrilateral mesh configurations
    mesh_sizes = [(80, 40), (120, 60), (160, 80), (200, 100), (240, 120),
                 (280, 140), (320, 160), (360, 180), (400, 200), (440, 220),
                 (480, 240)]
    
    c_values_quad = [8/size[0] for size in mesh_sizes]
    
    # DOFs for quadrilateral meshes (calculated based on mesh sizes)
    dofs_quad = [411522, 642402, 924482, 1257762, 1642242, 2077922, 
                2564802, 3102882, 3692162, 4332642, 5024322]
    
    # Adaptive mesh configurations
    config = ElasticWaterConfig()
    min_cell_sizes = config.min_cell_sizes
    dofs_adaptive = [492202, 354486, 492202, 600346]
    
    return c_values_quad, dofs_quad, min_cell_sizes, dofs_adaptive


def plot_data_preview_elastic(data_dict: Dict[str, np.ndarray], 
                             reference_data: np.ndarray) -> None:
    """
    Create preview plots of the simulation data.
    
    Args:
        data_dict: Dictionary containing all simulation data
        reference_data: Reference solution data
    """
    # Plot component 0
    plt.figure(figsize=(10, 6))
    
    # Plot selected cases
    if 'quad_11' in data_dict:
        plt.plot(data_dict['quad_11'][:, 0, 0], label="Case 11")
    plt.plot(reference_data[:, 0, 0], label="Reference")
    
    # Plot adaptive meshes
    adaptive_labels = ['Adaptive', 'Adaptive1', 'Adaptive2']
    adaptive_keys = ['adaptive', 'adaptive1', 'adaptive2']
    for key, label in zip(adaptive_keys, adaptive_labels):
        if key in data_dict:
            plt.plot(data_dict[key][:, 0, 0], label=label)
    
    plt.legend()
    plt.title("Component 0 Comparison")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot component 1
    plt.figure(figsize=(10, 6))
    
    if 'quad_11' in data_dict:
        plt.plot(data_dict['quad_11'][:, 0, 1], label="Case 11")
    plt.plot(reference_data[:, 0, 1], label="Reference")
    
    for key, label in zip(adaptive_keys, adaptive_labels):
        if key in data_dict:
            plt.plot(data_dict[key][:, 0, 1], label=label)
    
    plt.legend()
    plt.title("Component 1 Comparison")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.show()


def create_elastic_error_plot_improved(quad_errors: List[float], 
                                      adaptive_errors: List[float]) -> None:
    """
    Create improved publication-quality error plot for elastic water.
    
    Args:
        quad_errors: List of errors for quadrilateral meshes
        adaptive_errors: List of errors for adaptive meshes
    """
    # Get mesh configurations
    c_values_quad, dofs_quad, min_cell_sizes, dofs_adaptive = (
        get_elastic_mesh_configurations())
    
    # Use subset of quadrilateral data (cases 3-10, indices 2-9)
    quad_subset_indices = list(range(2, 10))  # Cases 3-10
    c_values_subset = [c_values_quad[i] for i in quad_subset_indices]
    errors_subset = [quad_errors[i] for i in quad_subset_indices 
                    if quad_errors[i] is not None]
    dofs_subset = [dofs_quad[i] for i in quad_subset_indices]
    
    # Filter out cases starting from index 2 (case 3)
    c_values_plot = c_values_subset[2:]  # Start from case 5
    errors_plot = errors_subset[2:]
    dofs_plot = dofs_subset[2:]
    
    # Use high-quality style
    plt.style.use("seaborn-v0_8-paper")
    
    # Create figure with improved proportions
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors and markers for better distinction
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple']
    markers = ['s', 'o', '*', '^', 'D']
    
    # Plot quadrilateral mesh results
    ax.plot(c_values_plot, errors_plot, 
           marker=markers[0], markersize=8, linestyle="-", linewidth=2,
           label="Quadrilateral mesh", color=colors[0])
    
    # Annotate quadrilateral points with DOFs (formatted with commas)
    for x, y, dof in zip(c_values_plot, errors_plot, dofs_plot):
        ax.annotate(f"{dof:,}", (x, y), fontsize=14, fontweight="bold", 
                   xytext=(-12, 7), textcoords="offset points", 
                   color=colors[0])
    
    # Plot adaptive mesh results
    adaptive_labels = ["Adaptive mesh (C = 3.0)", "Adaptive mesh (C = 2.7)", 
                      "Adaptive mesh (C = 2.3)", "Adaptive mesh (C = 2.0)"]
    
    for i, (error, label) in enumerate(zip(adaptive_errors[:3], 
                                          adaptive_labels[:3])):
        if error is not None:
            ax.plot([min_cell_sizes[i]], [error], 
                   marker=markers[i+1], markersize=10,
                   label=label, color=colors[i+1], linestyle="")
            
            # Annotate with DOF count (formatted with commas)
            ax.annotate(f"{dofs_adaptive[i]:,}", 
                       (min_cell_sizes[i], error), 
                       fontsize=14, fontweight="bold",
                       xytext=(0, -15), textcoords="offset points", 
                       ha="center", color=colors[i+1])
    
    # Detect if log scale would be beneficial
    all_errors = errors_plot + [e for e in adaptive_errors[:3] if e is not None]
    error_range = max(all_errors) / min(all_errors)
    if error_range > 100:
        ax.set_yscale('log')
        print("Applied log scale to y-axis due to large error range")
    
    # Labels and formatting
    ax.set_xlabel(r"Cell facet size (h)", fontsize=18)
    ax.set_ylabel(r"Relative Error", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=16)
    
    # Enhanced legend with multiple columns if needed and larger font
    legend = ax.legend(fontsize=16, loc="upper left", frameon=True)
    if len(legend.get_texts()) > 4:
        legend._ncol = 2
    
    # Grid for readability
    ax.grid(True, linestyle="--", alpha=0.6)
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    # Save as high-resolution files for publication
    output_formats = [
        ("error_plot_elastic_water.eps", "EPS"),
        ("error_plot_elastic_water.png", "PNG")
    ]
    
    for filename, format_name in output_formats:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved {format_name} plot: {filename}")
    
    # Show plot
    plt.show()


def main() -> None:
    """Main execution function."""
    print("Starting elastic water error computation...")
    
    try:
        # Load simulation data
        print("\n1. Loading simulation data...")
        data_dict, reference_data = load_elastic_simulation_data()
        
        # Compute errors
        print("\n2. Computing errors...")
        quad_errors, adaptive_errors = compute_elastic_errors(data_dict, 
                                                             reference_data)
        
        print(f"Computed {len([e for e in quad_errors if e is not None])} "
              f"quadrilateral errors")
        print(f"Computed {len([e for e in adaptive_errors if e is not None])} "
              f"adaptive errors")
        
        # Create data preview plots
        print("\n3. Creating data preview plots...")
        plot_data_preview_elastic(data_dict, reference_data)
        
        # Create improved error plot
        print("\n4. Creating improved error plot...")
        create_elastic_error_plot_improved(quad_errors, adaptive_errors)
        
        print("\nElastic water error computation completed successfully!")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        raise


if __name__ == "__main__":
    main()
