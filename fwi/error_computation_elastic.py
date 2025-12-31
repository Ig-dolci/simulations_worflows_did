"""
Improved elastic error computation script.

This script computes and visualizes errors for elastic wave simulations
with better code organization and publication-quality plots.
"""

import numpy as np
import matplotlib.pyplot as plt

# Configure matplotlib for LaTeX rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def load_elastic_data():
    """Load all elastic simulation data."""
    data_dir = "comp_performance"
    data = {}
    
    # Load quadrilateral mesh data
    for i in range(1, 12):
        filename = f"{data_dir}/rec_data_case{i}_elastic.npy"
        data[f'case_{i}'] = np.load(filename)
    
    # Load reference and adaptive data
    data['reference'] = np.load(f"{data_dir}/rec_data_reference_elastic.npy")
    data['adaptive'] = np.load(f"{data_dir}/rec_data_adapt_elastic_solid.npy")
    data['adaptive1'] = np.load(f"{data_dir}/rec_data_adapt_elastic_solid1.npy")
    data['adaptive2'] = np.load(f"{data_dir}/rec_data_adapt_elastic_solid2.npy")
    data['adaptive3'] = np.load(f"{data_dir}/rec_data_adapt_elastic_solid3.npy")
    
    return data


def compute_errors(data):
    """Compute relative errors for all cases."""
    reference = data['reference']
    reference_norm = np.linalg.norm(reference)
    errors = {}
    
    # Quadrilateral mesh errors
    for i in range(1, 12):
        case_data = data[f'case_{i}']
        errors[f'case_{i}'] = (np.linalg.norm(case_data - reference) / 
                               reference_norm)
    
    # Adaptive mesh errors
    for key in ['adaptive', 'adaptive1', 'adaptive2', 'adaptive3']:
        adaptive_data = data[key]
        errors[key] = (np.linalg.norm(adaptive_data - reference) / 
                       reference_norm)
    
    return errors


def create_data_preview_plots(data):
    """Create preview plots of the data."""
    # Plot component 0
    plt.figure(figsize=(12, 6))
    plt.plot(data['case_11'][:, 0, 0], label="Case 11")
    plt.plot(data['reference'][:, 0, 0], label="Reference")
    plt.plot(data['adaptive'][:, 0, 0], label="Adaptive")
    plt.plot(data['adaptive1'][:, 0, 0], label="Adaptive1")
    plt.plot(data['adaptive2'][:, 0, 0], label="Adaptive2")
    plt.legend(fontsize=14)
    plt.title("Component 0 Comparison", fontsize=16)
    plt.xlabel("Time Sample", fontsize=14)
    plt.ylabel("Amplitude", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot component 1
    plt.figure(figsize=(12, 6))
    plt.plot(data['case_11'][:, 0, 1], label="Case 11")
    plt.plot(data['reference'][:, 0, 1], label="Reference")
    plt.plot(data['adaptive'][:, 0, 1], label="Adaptive")
    plt.plot(data['adaptive1'][:, 0, 1], label="Adaptive1")
    plt.plot(data['adaptive2'][:, 0, 1], label="Adaptive2")
    plt.legend(fontsize=14)
    plt.title("Component 1 Comparison", fontsize=16)
    plt.xlabel("Time Sample", fontsize=14)
    plt.ylabel("Amplitude", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_improved_error_plot(errors):
    """Create improved publication-quality error plot."""
    # Configuration data
    c_values = [8/160, 8/200, 8/240, 8/280, 8/320, 8/360, 8/400, 8/440]
    quad_errors = [errors[f'case_{i}'] for i in range(3, 11)]
    dofs = [924482, 1257762, 1642242, 2077922, 2564802, 3102882, 3692162, 4332642]
    
    # Adaptive mesh configuration
    C = [2.7, 2.3, 2.0]
    vs = 0.5
    f = 10.0
    min_cell_size = [vs / (f * c) for c in C]
    dofs_triangle = [1222134, 892888, 676862]
    adaptive_errors = [errors['adaptive'], errors['adaptive1'], errors['adaptive2']]
    
    # Use high-quality style
    plt.style.use("seaborn-v0_8-paper")
    
    # Create figure with improved proportions
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot quadrilateral mesh (skip first 2 points to match original)
    c_values_plot = c_values[2:]
    quad_errors_plot = quad_errors[2:]
    dofs_plot = dofs[2:]
    
    ax.plot(c_values_plot, quad_errors_plot, marker="s", markersize=8, 
            linestyle="-", linewidth=2, label="Quadrilateral mesh", 
            color="tab:blue")
    
    # Annotate quadrilateral points with DOFs
    for x, y, dof in zip(c_values_plot, quad_errors_plot, dofs_plot):
        ax.annotate(f"{dof:,}", (x, y), fontsize=14, fontweight="bold", 
                    xytext=(-12, 7), textcoords="offset points", 
                    color="tab:blue")
    
    # Plot adaptive mesh results
    colors = ["tab:red", "tab:green", "tab:orange"]
    markers = ["o", "*", "^"]
    labels = ["Adaptive mesh (C = 2.7)", "Adaptive mesh (C = 2.3)", 
              "Adaptive mesh (C = 2.0)"]
    
    for i, (size, error, dof, color, marker, label) in enumerate(
            zip(min_cell_size, adaptive_errors, dofs_triangle, 
                colors, markers, labels)):
        ax.plot([size], [error], marker=marker, markersize=10,
                label=label, color=color, linestyle="")
        
        ax.annotate(f"{dof:,}", (size, error), fontsize=14, fontweight="bold",
                    xytext=(0, -15), textcoords="offset points", 
                    ha="center", color=color)
    
    # Formatting with larger fonts
    ax.set_xlabel(r"Cell facet size (h)", fontsize=18)
    ax.set_ylabel(r"Relative Error $\|\mathbf{u} - \mathbf{u}_{\mathrm{ref}}\| / \|\mathbf{u}_{\mathrm{ref}}\|$",
                  labelpad=10, fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.legend(fontsize=16, loc="upper left", frameon=True)
    
    # Grid for readability
    ax.grid(True, linestyle="--", alpha=0.6)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save as high-resolution EPS and PNG
    plt.savefig("error_plot_elastic_improved.eps", dpi=300, bbox_inches="tight")
    plt.savefig("error_plot_elastic_improved.png", dpi=300, bbox_inches="tight")
    print("Saved plots: error_plot_elastic_improved.eps and .png")
    
    plt.show()


# Main execution
print("Loading elastic simulation data...")
all_data = load_elastic_data()

print("Computing errors...")
all_errors = compute_errors(all_data)

print("Creating data preview plots...")
create_data_preview_plots(all_data)

print("Creating improved error plot...")
create_improved_error_plot(all_errors)

print("Analysis complete!")
