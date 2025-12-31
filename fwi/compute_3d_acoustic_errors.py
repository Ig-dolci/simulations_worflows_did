"""
Error computation and visualization for 3D acoustic wave propagation simulations.

This script computes relative errors between simulation results and reference
data for both hexahedral and tetrahedral meshes.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
from dataclasses import dataclass
import argparse

# Configure matplotlib for publication-quality plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


@dataclass
class SimulationConfig:
    """Configuration parameters for 3D simulation analysis."""
    vp_min: float = 1.5  # Minimum P-wave velocity (km/s)
    vp_max: float = 4.482  # Maximum P-wave velocity (km/s)
    frequency: float = 2.0  # Hz, dominant frequency
    domain_x: float = 13520.0  # Domain size in x (meters)
    domain_y: float = 13520.0  # Domain size in y (meters)
    domain_z: float = 4200.0  # Domain size in z (meters)


def load_simulation_data(
    mesh_type: str,
    cases: List[int]
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Load 3D simulation data from .npy files.
    Uses hexahedral reference (80x80x40) for all comparisons.

    Args:
        mesh_type: "hex" or "tet"
        cases: List of case numbers to load

    Returns:
        Tuple of (data_dict, reference_data)
    """
    data_dir = "comp_performance"

    if mesh_type == "hex":
        prefix = "rec_data_3d_hex_acoustic_"
    else:  # tet
        prefix = "rec_data_3d_tet_acoustic_"

    # Always use hexahedral reference (180x180x90)
    ref_file = f"{data_dir}/rec_data_3d_hex_acoustic_ref.npy"

    # Load case data
    case_data = {}
    for case_num in cases:
        filename = f"{data_dir}/{prefix}{case_num}.npy"
        if os.path.exists(filename):
            data = np.load(filename, allow_pickle=True)
            case_data[case_num] = data
            print(f"Loaded {filename}, shape: {np.array(data).shape}, dtype: {np.array(data).dtype}")
        else:
            print(f"Warning: {filename} not found")

    # Load reference data (always from hexahedral mesh)
    if os.path.exists(ref_file):
        reference_data = np.load(ref_file, allow_pickle=True)
        print(f"Loaded reference: {ref_file} (180x180x90 hexahedral mesh)")
        print(f"Reference shape: {np.array(reference_data).shape}, dtype: {np.array(reference_data).dtype}")
    else:
        print(f"Warning: Reference file {ref_file} not found")
        print("Please run reference case first to generate reference data")
        raise FileNotFoundError(f"Reference data not found: {ref_file}")

    return case_data, reference_data


def compute_relative_error(data: np.ndarray, reference: np.ndarray) -> float:
    """Compute relative L2 error between data and reference."""
    # Convert to arrays first
    data_arr = np.array(data)
    ref_arr = np.array(reference)
    
    # Check if data is array of arrays (time steps)
    if data_arr.ndim == 2:
        # Shape is (time_steps, receivers)
        data_flat = data_arr.flatten()
    elif data_arr.dtype == object:
        # Data is list of arrays, concatenate them
        data_flat = np.concatenate([np.array(d).flatten() for d in data_arr])
    else:
        data_flat = data_arr.flatten()
    
    # Same for reference
    if ref_arr.ndim == 2:
        ref_flat = ref_arr.flatten()
    elif ref_arr.dtype == object:
        ref_flat = np.concatenate([np.array(r).flatten() for r in ref_arr])
    else:
        ref_flat = ref_arr.flatten()
    
    # Check dimensions match
    if len(data_flat) != len(ref_flat):
        print(f"Warning: data shape {data_flat.shape} != reference shape {ref_flat.shape}")
        # Truncate to minimum length
        min_len = min(len(data_flat), len(ref_flat))
        data_flat = data_flat[:min_len]
        ref_flat = ref_flat[:min_len]
    
    if len(ref_flat) == 0:
        raise ValueError("Reference data is empty!")
    
    return np.linalg.norm(data_flat - ref_flat) / np.linalg.norm(ref_flat)


def get_hex_mesh_configurations() -> Tuple[List[int], List[int]]:
    """
    Get mesh configurations for hexahedral meshes.

    Returns:
        Tuple of (case_numbers, dofs)
    """
    # Hexahedral mesh configurations (degree=3 spectral elements)
    # Updated mesh refinement: starting at 40x40x20
    hex_configs = [
        (1, 40, 40, 20),
        (2, 50, 50, 25),
        (3, 60, 60, 30),
        (4, 70, 70, 35),
        (5, 80, 80, 40),
        (6, 90, 90, 45),
        (7, 100, 100, 50),
        (8, 110, 110, 55),
        (9, 120, 120, 60),
        (10, 130, 130, 65),
        (11, 140, 140, 70),
        (12, 150, 150, 75),
        (13, 160, 160, 80),
    ]
    
    # DOFs = (elem_x + 1) * (elem_y + 1) * (elem_z + 1)
    dofs = [(ex + 1) * (ey + 1) * (ez + 1) for _, ex, ey, ez in hex_configs]
    case_numbers = [c for c, _, _, _ in hex_configs]
    
    return case_numbers, dofs


def get_tet_mesh_configurations() -> Tuple[List[int], List[int]]:
    """
    Get mesh configurations for tetrahedral meshes.

    Returns:
        Tuple of (case_numbers, dofs)
    """
    # Tetrahedral mesh configurations (degree=3 KMV elements)
    # Updated mesh refinement: starting at 40x40x20
    tet_configs = [
        (1, 40, 40, 20),
        (2, 50, 50, 25),
        (3, 60, 60, 30),
        (4, 70, 70, 35),
        (5, 80, 80, 40),
        (6, 90, 90, 45),
        (7, 100, 100, 50),
        (8, 110, 110, 55),
        (9, 120, 120, 60),
        (10, 130, 130, 65),
        (11, 140, 140, 70),
        (12, 150, 150, 75),
        (13, 160, 160, 80),
    ]
    
    # For tetrahedral meshes with degree=3, DOFs are approximately:
    # DOFs ≈ 6 * (elem_x + 1) * (elem_y + 1) * (elem_z + 1) (rough estimate)
    # These will need to be updated with actual values from simulations
    dofs = [6 * (ex + 1) * (ey + 1) * (ez + 1) for _, ex, ey, ez in tet_configs]
    case_numbers = [c for c, _, _, _ in tet_configs]
    
    return case_numbers, dofs


def compute_all_errors(
    case_data: Dict[int, np.ndarray],
    reference_data: np.ndarray,
    cases: List[int]
) -> List[float]:
    """
    Compute errors for all cases.

    Returns:
        List of errors for each case
    """
    errors = []
    for case_num in cases:
        if case_num in case_data:
            error = compute_relative_error(case_data[case_num], reference_data)
            errors.append(error)
            print(f"Case {case_num}: Error = {error:.6e}")
        else:
            errors.append(np.nan)
            print(f"Case {case_num}: Data not found")
    
    return errors


def plot_data_preview(
    case_data: Dict[int, np.ndarray],
    reference_data: np.ndarray,
    mesh_type: str,
    cases_to_plot: List[int]
) -> None:
    """Create a preview plot of selected simulation data."""
    plt.figure(figsize=(12, 6))
    
    # Plot selected cases
    for case_num in cases_to_plot:
        if case_num in case_data:
            data = np.array(case_data[case_num])
            if isinstance(data[0], np.ndarray):
                # If data is list of arrays, concatenate
                data = np.concatenate(data)
            plt.plot(data, label=f"Case {case_num}", linewidth=1.5, alpha=0.7)
    
    # Plot reference
    ref_data = np.array(reference_data)
    if isinstance(ref_data[0], np.ndarray):
        ref_data = np.concatenate(ref_data)
    plt.plot(ref_data, label="Reference (80x80x40)", linewidth=2, color="black")
    
    plt.legend(fontsize=11)
    plt.xlabel("Time samples", fontsize=13)
    plt.ylabel("Amplitude", fontsize=13)
    plt.title(f"3D Acoustic Simulation Data - {mesh_type.upper()} Mesh", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"preview_3d_{mesh_type}_mesh.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Preview plot saved as preview_3d_{mesh_type}_mesh.png")


def create_error_comparison_plot(
    hex_dofs: List[int],
    hex_errors: List[float],
    tet_dofs: List[int] = None,
    tet_errors: List[float] = None,
    output_filename: str = "error_comparison_3d_hex_tet.png"
) -> None:
    """Create error comparison plot for hex and tet meshes.
    All errors are computed against hexahedral reference (80x80x40)."""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    
    # Plot: Error vs DOFs
    ax.plot(
        hex_dofs, hex_errors,
        marker="s", markersize=12, linestyle="-", linewidth=3,
        label="Hexahedral mesh (degree=1)", color="tab:blue"
    )
    
    # Plot tetrahedral mesh if data is provided
    if tet_dofs is not None and tet_errors is not None:
        ax.plot(
            tet_dofs, tet_errors,
            marker="^", markersize=12, linestyle="-", linewidth=3,
            label="Tetrahedral mesh (degree=3)", color="tab:red"
        )
    
    # Annotate some key points with DOFs
    for i in [0, len(hex_dofs)//2, -1]:
        if i < len(hex_dofs):
            ax.annotate(
                f"{hex_dofs[i]:,}",
                (hex_dofs[i], hex_errors[i]),
                fontsize=11, xytext=(5, 5), textcoords="offset points",
                color="tab:blue", fontweight='bold'
            )
    
    if tet_dofs is not None and tet_errors is not None:
        for i in [0, len(tet_dofs)//2, -1]:
            if i < len(tet_dofs):
                ax.annotate(
                    f"{tet_dofs[i]:,}",
                    (tet_dofs[i], tet_errors[i]),
                    fontsize=11, xytext=(5, -15), textcoords="offset points",
                    color="tab:red", fontweight='bold'
                )
    
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=1.2)
    ax.set_xlabel(r"Degrees of Freedom (DOFs)", fontsize=18, fontweight='bold')
    ax.set_ylabel(r"$\frac{\|u - u_{\mathrm{ref}}\|_2}{\|u_{\mathrm{ref}}\|_2}$", fontsize=20)
    ax.set_title("3D Acoustic Wave Convergence Analysis", fontsize=20, fontweight='bold', pad=20)
    ax.legend(fontsize=15, loc='best', frameon=True, shadow=True, fancybox=True)
    ax.tick_params(axis="both", labelsize=14, width=1.5, length=6)
    
    # Use log-log scale to show convergence order as straight line
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add more tick marks on both axes
    from matplotlib.ticker import LogLocator, NullFormatter
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())
    
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())
    
    # Enhance spine appearance
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Save plots in high quality
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.savefig(output_filename.replace('.png', '.eps'),
                format='eps', bbox_inches="tight")
    plt.savefig(output_filename.replace('.png', '.pdf'),
                format='pdf', bbox_inches="tight")
    
    print(f"Comparison plot saved as {output_filename}")
    print(f"Also saved as .eps and .pdf formats")
    plt.show()


def main():
    """Main function to execute the analysis."""
    parser = argparse.ArgumentParser(
        description='Compute and plot errors for 3D acoustic simulations'
    )
    parser.add_argument(
        '--start-case', type=int, default=1,
        help='Starting case number (default: 1)'
    )
    parser.add_argument(
        '--end-case', type=int, default=13,
        help='Ending case number (default: 13)'
    )
    parser.add_argument(
        '--preview', action='store_true',
        help='Show data preview plots'
    )
    parser.add_argument(
        '--hex-only', action='store_true',
        help='Only analyze hexahedral meshes'
    )
    args = parser.parse_args()
    
    cases = list(range(args.start_case, args.end_case + 1))
    
    print("="*60)
    print("Error Computation for 3D Acoustic Simulations")
    print("Reference: Hexahedral mesh 80x80x40 (degree=1)")
    print("="*60)
    
    # Load hexahedral mesh data
    print("\n--- Loading Hexahedral Mesh Data ---")
    hex_data, hex_reference = load_simulation_data("hex", cases)
    
    # Load tetrahedral mesh data if requested
    tet_data = None
    tet_reference = None
    if not args.hex_only:
        print("\n--- Loading Tetrahedral Mesh Data ---")
        tet_data, tet_reference = load_simulation_data("tet", cases)
        print("\nNote: Both hex and tet errors are computed against")
        print("      the hexahedral 80x80x40 reference mesh")
    
    # Get mesh configurations
    print("\n--- Getting Mesh Configurations ---")
    hex_cases, hex_dofs = get_hex_mesh_configurations()
    
    # Filter to only cases we have data for
    hex_cases_filtered = [c for c in hex_cases if c in cases and c in hex_data]
    
    # Compute errors
    print("\n--- Computing Errors ---")
    print("\nHexahedral Mesh Errors:")
    hex_errors = compute_all_errors(hex_data, hex_reference, hex_cases_filtered)
    
    # Filter DOFs for cases we have
    hex_idx = [hex_cases.index(c) for c in hex_cases_filtered]
    hex_dofs_filtered = [hex_dofs[i] for i in hex_idx]
    
    # Compute tetrahedral mesh errors if data available
    tet_dofs_filtered = None
    tet_errors = None
    if tet_data is not None:
        tet_cases, tet_dofs = get_tet_mesh_configurations()
        tet_cases_filtered = [c for c in tet_cases if c in cases and c in tet_data]
        
        if tet_cases_filtered:
            print("\nTetrahedral Mesh Errors:")
            tet_errors = compute_all_errors(tet_data, tet_reference, tet_cases_filtered)
            
            tet_idx = [tet_cases.index(c) for c in tet_cases_filtered]
            tet_dofs_filtered = [tet_dofs[i] for i in tet_idx]
        else:
            print("\nWarning: No tetrahedral mesh data found for requested cases")
            print("Run ./run_3d_acoustic_tet_cases.sh first")
    
    # Create preview plots if requested
    if args.preview:
        print("\n--- Creating Preview Plots ---")
        preview_cases = [args.start_case, (args.start_case + args.end_case)//2, args.end_case]
        plot_data_preview(hex_data, hex_reference, "hex", preview_cases)
        if tet_data is not None:
            plot_data_preview(tet_data, tet_reference, "tet", preview_cases)
    
    # Create comparison plot
    print("\n--- Creating Error Comparison Plot ---")
    create_error_comparison_plot(
        hex_dofs_filtered, hex_errors,
        tet_dofs_filtered, tet_errors
    )
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
