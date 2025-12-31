"""
Improved elastic error computation script with enhanced code quality.

This script computes and visualizes errors for elastic wave simulations using:
- Dynamic data loading functions
- Modular, reusable code structure
- Publication-quality plots with larger fonts
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
class ElasticConfig:
    """Configuration for elastic wave simulations."""
    vs: float = 0.5  # S-wave velocity
    vp: float = 1.5  # P-wave velocity
    frequency: float = 10.0  # Hz
    c_values: List[float] = None
    domain_size: float = 8.0
    
    def __post_init__(self):
        if self.c_values is None:
            self.c_values = [2.7, 2.3, 2.0]
    
    @property
    def min_cell_sizes(self) -> List[float]:
        """Calculate minimum cell size for each C value."""
        return [self.vs / (self.frequency * c) for c in self.c_values]


@dataclass
class MeshDataElastic:
    """Container for elastic mesh data and configuration."""
    quad_data: Dict[int, np.ndarray]
    adaptive_data: Dict[str, np.ndarray]
    reference_data: np.ndarray
    quad_errors: List[float]
    adaptive_errors: List[float]
    mesh_configurations: Tuple[List[float], List[int], List[float], List[int]]


def load_elastic_simulation_data() -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Load all elastic simulation data dynamically.
    
    Returns:
        Tuple containing dictionaries of simulation data and reference data
    """
    data_dict = {}
    data_dir = "comp_performance"
    
    # Load quadrilateral mesh data (cases 1-11)
    for i in range(1, 12):
        filename = f"{data_dir}/rec_data_case{i}_elastic.npy"
        if os.path.exists(filename):
            data_dict[f'quad_{i}'] = np.load(filename)
            print(f"Loaded {filename}")
        else:
            print(f"Warning: {filename} not found")
    
    # Load adaptive mesh data
    adaptive_files = [
        ('adaptive', f'{data_dir}/rec_data_adapt_elastic_solid.npy'),
        ('adaptive1', f'{data_dir}/rec_data_adapt_elastic_solid1.npy'),
        ('adaptive2', f'{data_dir}/rec_data_adapt_elastic_solid2.npy'),
        ('adaptive3', f'{data_dir}/rec_data_adapt_elastic_solid3.npy')
    ]
    
    for key, filename in adaptive_files:
        if os.path.exists(filename):
            data_dict[key] = np.load(filename)
            print(f"Loaded {filename}")
        else:
            print(f"Warning: {filename} not found")
    
    # Load reference data
    reference_filename = f"{data_dir}/rec_data_reference_elastic.npy"
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
    Compute relative errors for all elastic simulation cases.
    
    Args:
        data_dict: Dictionary containing all simulation data
        reference_data: Reference solution data
    
    Returns:
        Tuple of (quadrilateral errors, adaptive errors)
    """
    reference_norm = np.linalg.norm(reference_data)
    
    # Compute errors for quadrilateral meshes (cases 1-11)
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
    Get mesh configuration data for elastic simulations.
    
    Returns:
        Tuple of (c_values for quads, dofs for quads,
                 min_cell_sizes for adaptive, dofs for adaptive)
    """
    # Quadrilateral mesh configurations
    # Mesh sizes: (nx, ny) for cases 1-11
    mesh_sizes = [(80, 40), (120, 60), (160, 80), (200, 100), (240, 120),
                  (280, 140), (320, 160), (360, 180), (400, 200), (440, 220),
                  (480, 240)]
    
    # Calculate cell sizes (domain_size / nx)
    domain_size = 8.0
    c_values_quad = [domain_size / size[0] for size in mesh_sizes]
    
    # DOFs for quadrilateral meshes (2 components for elastic)
    dofs_quad = [411522, 642402, 924482, 1257762, 1642242, 2077922,
                 2564802, 3102882, 3692162, 4332642, 5024322]
    
    # Adaptive mesh configurations
    config = ElasticConfig()
    min_cell_sizes = config.min_cell_sizes
    dofs_adaptive = [1222134, 892888, 676862, 0]  # Last one not used
    
    return c_values_quad, dofs_quad, min_cell_sizes, dofs_adaptive


def plot_data_preview_elastic(data_dict: Dict[str, np.ndarray],
                              reference_data: np.ndarray) -> None:
    """
    Create preview plots of the elastic simulation data.
    
    Args:
        data_dict: Dictionary containing all simulation data
        reference_data: Reference solution data
    """
    # Plot component 0 (x-displacement)
    plt.figure(figsize=(12, 6))
    
    # Plot selected cases
    if 'quad_11' in data_dict:
        plt.plot(data_dict['quad_11'][:, 0, 0], label="Case 11", linewidth=1.5)
    plt.plot(reference_data[:, 0, 0], label="Reference", linewidth=2,
             color="black")
    
    # Plot adaptive meshes
    adaptive_labels = ['Adaptive (C=2.7)', 'Adaptive1 (C=2.3)',
                       'Adaptive2 (C=2.0)']
    adaptive_keys = ['adaptive', 'adaptive1', 'adaptive2']
    colors = ['tab:red', 'tab:green', 'tab:orange']
    
    for key, label, color in zip(adaptive_keys, adaptive_labels, colors):
        if key in data_dict:
            plt.plot(data_dict[key][:, 0, 0], label=label, linestyle="--",
                     linewidth=1.5, color=color)
    
    plt.legend(fontsize=12)
    plt.title("Component 0 (x-displacement) Comparison", fontsize=14)
    plt.xlabel("Time Sample", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot component 1 (y-displacement)
    plt.figure(figsize=(12, 6))
    
    if 'quad_11' in data_dict:
        plt.plot(data_dict['quad_11'][:, 0, 1], label="Case 11", linewidth=1.5)
    plt.plot(reference_data[:, 0, 1], label="Reference", linewidth=2,
             color="black")
    
    for key, label, color in zip(adaptive_keys, adaptive_labels, colors):
        if key in data_dict:
            plt.plot(data_dict[key][:, 0, 1], label=label, linestyle="--",
                     linewidth=1.5, color=color)
    
    plt.legend(fontsize=12)
    plt.title("Component 1 (y-displacement) Comparison", fontsize=14)
    plt.xlabel("Time Sample", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_elastic_error_plot_improved(quad_errors: List[float],
                                       adaptive_errors: List[float]) -> None:
    """
    Create improved publication-quality error plot for elastic waves.
    
    Args:
        quad_errors: List of errors for quadrilateral meshes
        adaptive_errors: List of errors for adaptive meshes
    """
    # Get mesh configurations
    c_values_quad, dofs_quad, min_cell_sizes, dofs_adaptive = (
        get_elastic_mesh_configurations())
    
    # Use subset of quadrilateral data (cases 3-10, indices 2-9)
    # This matches the original code: c_values[2:] and errors[2:]
    quad_subset_start = 2  # Case 3
    quad_subset_end = 10   # Case 10
    
    c_values_plot = c_values_quad[quad_subset_start:quad_subset_end+1]
    errors_plot = [quad_errors[i] for i in range(quad_subset_start, 
                   quad_subset_end+1) if quad_errors[i] is not None]
    dofs_plot = dofs_quad[quad_subset_start:quad_subset_end+1]
    
    # Skip first 2 cases to match original: c_values[2:] becomes c_values[4:]
    c_values_final = c_values_plot[2:]
    errors_final = errors_plot[2:]
    dofs_final = dofs_plot[2:]
    
    # Use high-quality style
    plt.style.use("seaborn-v0_8-paper")
    
    # Create figure with improved proportions
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors and markers for better distinction
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']
    markers = ['s', 'o', '*', '^']
    
    # Plot quadrilateral mesh results
    ax.plot(c_values_final, errors_final,
            marker=markers[0], markersize=8, linestyle="-", linewidth=2,
            label="Quadrilateral mesh", color=colors[0])
    
    # Annotate quadrilateral points with DOFs (formatted with commas)
    for x, y, dof in zip(c_values_final, errors_final, dofs_final):
        ax.annotate(f"{dof:,}", (x, y), fontsize=14, fontweight="bold",
                    xytext=(-12, 7), textcoords="offset points",
                    color=colors[0])
    
    # Plot adaptive mesh results (only first 3)
    adaptive_labels = ["Adaptive mesh (C = 2.7)", "Adaptive mesh (C = 2.3)",
                       "Adaptive mesh (C = 2.0)"]
    
    for i, (error, label) in enumerate(zip(adaptive_errors[:3],
                                           adaptive_labels)):
        if error is not None and i < len(min_cell_sizes):
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
    all_errors = errors_final + [e for e in adaptive_errors[:3]
                                 if e is not None]
    if all_errors:
        error_range = max(all_errors) / min(all_errors)
        if error_range > 100:
            ax.set_yscale('log')
            print("Applied log scale to y-axis due to large error range")
    
    # Labels and formatting with larger fonts
    ax.set_xlabel(r"Cell facet size (h)", fontsize=18)
    ax.set_ylabel(r"Relative Error $\|u - u_{\mathrm{ref}}\| / \|u_{\mathrm{ref}}\|$", 
                  fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=16)
    
    # Enhanced legend with larger font
    legend = ax.legend(fontsize=16, loc="upper left", frameon=True)
    if len(legend.get_texts()) > 4:
        legend._ncol = 2
    
    # Grid for readability
    ax.grid(True, linestyle="--", alpha=0.6)
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    # Save as high-resolution files for publication
    output_formats = [
        ("error_plot_elastic_improved.eps", "EPS"),
        ("error_plot_elastic_improved.png", "PNG")
    ]
    
    for filename, format_name in output_formats:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved {format_name} plot: {filename}")
    
    # Show plot
    plt.show()


def main() -> None:
    """Main execution function."""
    print("Starting improved elastic error computation...")
    
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
        
        print("\nElastic error computation completed successfully!")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        raise


if __name__ == "__main__":
    main()
