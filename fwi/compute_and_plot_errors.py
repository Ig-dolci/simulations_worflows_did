"""
Error computation and visualization for acoustic wave propagation simulations.

This script computes relative errors between simulation results and reference
data for both quadrilateral and triangular meshes.
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
    """Configuration parameters for simulation analysis."""
    vs: float = 0.5  # S-wave velocity
    vp: float = 1.5  # P-wave velocity
    frequency: float = 10.0  # Hz, representative frequency
    domain_size: float = 8.0  # Domain size for mesh size calculation


def load_simulation_data(
    mesh_type: str,
    cases: List[int]
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Load simulation data from .npy files.
    Uses quadrilateral reference (540x270) for all comparisons.

    Args:
        mesh_type: "quad", "triangle", or "triangle_adaptive"
        cases: List of case numbers to load

    Returns:
        Tuple of (data_dict, reference_data)
    """
    data_dir = "comp_performance"

    if mesh_type == "quad":
        prefix = "rec_data_quad_no_adaptive_acoustic_"
    elif mesh_type == "triangle":
        prefix = "rec_data_triangle_no_adaptive_acoustic_"
    else:  # triangle_adaptive
        prefix = "rec_data_triangle_adaptive_acoustic_"

    # Always use quadrilateral reference (540x270)
    ref_file = f"{data_dir}/rec_data_quad_no_adaptive_acoustic_ref.npy"

    # Load case data
    case_data = {}
    for case_num in cases:
        filename = f"{data_dir}/{prefix}{case_num}.npy"
        if os.path.exists(filename):
            case_data[case_num] = np.load(filename)
            print(f"Loaded {filename}")
        else:
            print(f"Warning: {filename} not found")

    # Load reference data (always from quadrilateral mesh)
    if os.path.exists(ref_file):
        reference_data = np.load(ref_file)
        print(f"Loaded reference: {ref_file} (540x270 quadrilateral mesh)")
    else:
        print(f"Warning: Reference file {ref_file} not found")
        print(f"Please run ./run_reference_case.sh first to generate reference data")
        raise FileNotFoundError(f"Reference data not found: {ref_file}")

    return case_data, reference_data


def compute_relative_error(data: np.ndarray, reference: np.ndarray) -> float:
    """Compute relative L2 error between data and reference."""
    # Ensure both arrays are 1D
    data_flat = np.array(data).flatten()
    ref_flat = np.array(reference).flatten()
    
    # Handle case where data might be list of arrays
    if isinstance(data, list):
        data_flat = np.concatenate([np.array(d).flatten() for d in data])
    if isinstance(reference, list):
        ref_flat = np.concatenate([np.array(r).flatten() for r in reference])
    
    # Check if lengths match, if not truncate to minimum
    if len(data_flat) != len(ref_flat):
        min_len = min(len(data_flat), len(ref_flat))
        print(f"Warning: Truncating from {len(data_flat)} and {len(ref_flat)} to {min_len} samples")
        data_flat = data_flat[:min_len]
        ref_flat = ref_flat[:min_len]
    
    if len(ref_flat) == 0:
        raise ValueError("Reference data is empty!")
    
    return np.linalg.norm(data_flat - ref_flat) / np.linalg.norm(ref_flat)


def get_quad_mesh_configurations() -> Tuple[List[int], List[int]]:
    """
    Get mesh configurations for quadrilateral meshes.

    Returns:
        Tuple of (case_numbers, dofs)
    """
    # Quadrilateral mesh configurations (from cases 4-13)
    quad_configs = [
        (4, 200, 100),
        (5, 240, 120),
        (6, 280, 140),
        (7, 320, 160),
        (8, 360, 180),
        (9, 400, 200),
        (10, 440, 220),
        (11, 480, 240),
        (12, 500, 250),
        (13, 520, 260),
    ]
    dofs = [
        321201,
        462241,
        628881,
        821121,
        1038961,
        1280401,
        1551441,
        1846081,
        2003001,
        2166321
    ]
    
    case_numbers = [c for c, _, _ in quad_configs]
    
    return case_numbers, dofs


def get_triangle_mesh_configurations() -> Tuple[List[int], List[int]]:
    """
    Get mesh configurations for triangular meshes.

    Returns:
        Tuple of (case_numbers, dofs)
    """
    # Triangular mesh configurations (from cases 4-13)
    triangle_configs = [
        (4, 172, 85),
        (5, 205, 102),
        (6, 239, 119),
        (7, 274, 136),
        (8, 290, 145),
        (9, 340, 170),
        (10, 390, 180),
        (11, 418, 204),
        (12, 425, 214),
        (13, 444, 222),
    ]

    dofs = [
        322669,
        461249,
        627135,
        821449,
        926841,
        1273641,
        1546681,
        1878473,
        2003457,
        2171161
    ]
    
    case_numbers = [c for c, _, _ in triangle_configs]
    
    return case_numbers, dofs


def get_triangle_adaptive_mesh_configurations() -> Tuple[List[int], List[int]]:
    """
    Get mesh configurations for adaptive triangular meshes.

    Returns:
        Tuple of (case_numbers, dofs)
    """
    # Adaptive triangular mesh configurations
    # DOFs values from actual simulations - ordered by DOFs
    adaptive_configs = [
        (23, 89733),   # mm_acoustic_23.msh
        (27, 125083),  # mm_acoustic_27.msh
        (3, 152392),   # mm_acoustic_3.msh
        (33, 183665),  # mm_acoustic_33.msh
        (36, 217134),  # mm_acoustic_36.msh
        (4, 274818),   # mm_acoustic_4.msh
        (45, 344869),  # mm_acoustic_45.msh
        (5, 424014),   # mm_acoustic_5.msh
        (55, 513508),  # mm_acoustic_55.msh
        (6, 609783),   # mm_acoustic_6.msh
        (65, 719583),  # mm_acoustic_65.msh
        (7, 833937),   # mm_acoustic_7.msh
        (75, 958363),  # mm_acoustic_75.msh
    ]
    
    case_numbers = [c for c, _ in adaptive_configs]
    dofs = [d for _, d in adaptive_configs]
    
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
    plt.plot(ref_data, label="Reference", linewidth=2, color="black")
    
    plt.legend(fontsize=11)
    plt.xlabel("Time samples", fontsize=13)
    plt.ylabel("Amplitude", fontsize=13)
    plt.title(f"Simulation Data Preview - {mesh_type.capitalize()} Mesh", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"preview_{mesh_type}_mesh.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Preview plot saved as preview_{mesh_type}_mesh.png")


def create_error_comparison_plot(
    quad_dofs: List[int],
    quad_errors: List[float],
    tri_dofs: List[int],
    tri_errors: List[float],
    tri_adapt_dofs: List[int] = None,
    tri_adapt_errors: List[float] = None,
    output_filename: str = "error_comparison_quad_triangle.png"
) -> None:
    """Create error comparison plot for quad and triangle meshes.
    All errors are computed against quadrilateral reference (540x270)."""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    
    # Plot: Error vs DOFs
    ax.plot(
        quad_dofs, quad_errors,
        marker="s", markersize=12, linestyle="-", linewidth=3,
        label="Quadrilateral mesh", color="tab:blue"
    )
    ax.plot(
        tri_dofs, tri_errors,
        marker="^", markersize=12, linestyle="-", linewidth=3,
        label="Triangular mesh (uniform)", color="tab:red"
    )
    
    # Plot adaptive mesh if data is provided
    if tri_adapt_dofs is not None and tri_adapt_errors is not None:
        ax.plot(
            tri_adapt_dofs, tri_adapt_errors,
            marker="o", markersize=12, linestyle="-", linewidth=3,
            label="Triangular mesh (adaptive)", color="tab:green"
        )
    
    # Annotate some key points with DOFs
    for i in [0, len(quad_dofs)//2, -1]:
        if i < len(quad_dofs):
            ax.annotate(
                f"{quad_dofs[i]:,}",
                (quad_dofs[i], quad_errors[i]),
                fontsize=11, xytext=(5, 5), textcoords="offset points",
                color="tab:blue", fontweight='bold'
            )
    
    for i in [0, len(tri_dofs)//2, -1]:
        if i < len(tri_dofs):
            ax.annotate(
                f"{tri_dofs[i]:,}",
                (tri_dofs[i], tri_errors[i]),
                fontsize=11, xytext=(5, -15), textcoords="offset points",
                color="tab:red", fontweight='bold'
            )
    
    # Annotate adaptive mesh points with case numbers (23 = 2.3, 27 = 2.7, etc.)
    if tri_adapt_dofs is not None and tri_adapt_errors is not None:
        # Get the case numbers for adaptive meshes (ordered by DOFs)
        adaptive_case_numbers = [23, 27, 3, 33, 36, 4, 45, 5, 55, 6, 65, 7, 75]
        for i, (dof, error) in enumerate(zip(tri_adapt_dofs, tri_adapt_errors)):
            if i < len(adaptive_case_numbers):
                case_num = adaptive_case_numbers[i]
                # Format label: two digits = decimal (23=2.3), one digit = int
                if case_num >= 10:
                    label = f"{case_num//10}.{case_num % 10}"
                else:
                    label = f"{case_num}"
                
                ax.annotate(
                    label,
                    (dof, error),
                    fontsize=10, xytext=(5, 5), textcoords="offset points",
                    color="tab:green", fontweight="bold"
                )
    
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=1.2)
    ax.set_xlabel(r"Degrees of Freedom (DOFs)", fontsize=18, fontweight='bold')
    ax.set_ylabel(r"$\frac{\|u - u_{\mathrm{ref}}\|_2}{\|u_{\mathrm{ref}}\|_2}$", fontsize=20)
    ax.set_title("Convergence Analysis", fontsize=20, fontweight='bold', pad=20)
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
        description='Compute and plot errors for acoustic simulations'
    )
    parser.add_argument(
        '--start-case', type=int, default=4,
        help='Starting case number (default: 4)'
    )
    parser.add_argument(
        '--end-case', type=int, default=13,
        help='Ending case number (default: 13)'
    )
    parser.add_argument(
        '--preview', action='store_true',
        help='Show data preview plots'
    )
    args = parser.parse_args()
    
    cases = list(range(args.start_case, args.end_case + 1))
    adaptive_cases = [3, 4, 5, 6, 7, 23, 27, 33, 36, 45, 55, 65, 75]
    
    print("="*60)
    print("Error Computation and Plotting for Acoustic Simulations")
    print("Reference: Quadrilateral mesh 540x270")
    print("="*60)
    
    # Load quadrilateral mesh data
    print("\n--- Loading Quadrilateral Mesh Data ---")
    quad_data, quad_reference = load_simulation_data("quad", cases)
    
    # Load triangular mesh data
    print("\n--- Loading Triangular Mesh Data ---")
    tri_data, tri_reference = load_simulation_data("triangle", cases)
    
    print("\nNote: Both quad and triangle errors are computed against")
    print("      the quadrilateral 540x270 reference mesh")
    
    # Load adaptive mesh data if requested
    tri_adapt_data = None
    tri_adapt_reference = None
    print("\n--- Loading Adaptive Triangular Mesh Data ---")
    tri_adapt_data, tri_adapt_reference = load_simulation_data(
        "triangle_adaptive", adaptive_cases
    )
    
    # Get mesh configurations
    print("\n--- Getting Mesh Configurations ---")
    quad_cases, quad_dofs = get_quad_mesh_configurations()
    tri_cases, tri_dofs = get_triangle_mesh_configurations()
    tri_adapt_cases, tri_adapt_dofs = get_triangle_adaptive_mesh_configurations()
    
    # Filter to only cases we have data for
    quad_cases_filtered = [c for c in quad_cases if c in cases and c in quad_data]
    tri_cases_filtered = [c for c in tri_cases if c in cases and c in tri_data]
    
    # Filter quad cases to keep only those with DOFs close to triangle cases
    # Get triangle DOFs range
    tri_dofs_filtered_initial = [tri_dofs[tri_cases.index(c)] for c in tri_cases_filtered]
    min_tri_dof = min(tri_dofs_filtered_initial)
    max_tri_dof = max(tri_dofs_filtered_initial)
    
    # Keep only quad cases within or near triangle DOF range (±20% margin)
    margin = 0.2
    quad_cases_filtered_final = []
    for case in quad_cases_filtered:
        idx = quad_cases.index(case)
        dof = quad_dofs[idx]
        if min_tri_dof * (1 - margin) <= dof <= max_tri_dof * (1 + margin):
            quad_cases_filtered_final.append(case)
    
    print(f"\nFiltered quad cases to match triangle DOF range:")
    print(f"Triangle DOFs range: {min_tri_dof:,} - {max_tri_dof:,}")
    print(f"Kept quad cases: {quad_cases_filtered_final}")
    
    quad_cases_filtered = quad_cases_filtered_final
    
    # Compute errors
    print("\n--- Computing Errors ---")
    print("\nQuadrilateral Mesh Errors:")
    quad_errors = compute_all_errors(quad_data, quad_reference, quad_cases_filtered)
    
    print("\nTriangular Mesh Errors:")
    tri_errors = compute_all_errors(tri_data, tri_reference, tri_cases_filtered)
    
    # Compute adaptive mesh errors if requested
    tri_adapt_errors = None
    tri_adapt_dofs_filtered = None
    if tri_adapt_data is not None:
        print("\nAdaptive Triangular Mesh Errors:")
        print(f"Requested adaptive cases: {adaptive_cases}")
        print(f"Available adaptive cases: {list(tri_adapt_data.keys())}")
        
        tri_adapt_cases_filtered = [
            c for c in tri_adapt_cases if c in adaptive_cases and c in tri_adapt_data
        ]
        print(f"Filtered adaptive cases: {tri_adapt_cases_filtered}")
        
        if tri_adapt_cases_filtered:
            tri_adapt_errors = compute_all_errors(
                tri_adapt_data, tri_adapt_reference, tri_adapt_cases_filtered
            )
            # Filter DOFs for cases we have
            tri_adapt_idx = [tri_adapt_cases.index(c) for c in tri_adapt_cases_filtered]
            tri_adapt_dofs_filtered = [tri_adapt_dofs[i] for i in tri_adapt_idx]
            print(f"Adaptive DOFs: {tri_adapt_dofs_filtered}")
            print(f"Adaptive errors: {tri_adapt_errors}")
        else:
            print("Warning: No adaptive mesh data found for requested cases")
            print("Make sure to run ./run_triangle_adaptive_cases.sh first")
    
    # Filter mesh sizes and cell sizes for cases we have
    quad_idx = [quad_cases.index(c) for c in quad_cases_filtered]
    tri_idx = [tri_cases.index(c) for c in tri_cases_filtered]
    
    quad_dofs_filtered = [quad_dofs[i] for i in quad_idx]
    tri_dofs_filtered = [tri_dofs[i] for i in tri_idx]
    
    # Create preview plots if requested
    if args.preview:
        print("\n--- Creating Preview Plots ---")
        preview_cases = [args.start_case, (args.start_case + args.end_case)//2, args.end_case]
        plot_data_preview(quad_data, quad_reference, "quad", preview_cases)
        plot_data_preview(tri_data, tri_reference, "triangle", preview_cases)
        if tri_adapt_data is not None and len(adaptive_cases) > 0:
            adaptive_preview = [adaptive_cases[0], adaptive_cases[-1]]
            plot_data_preview(tri_adapt_data, tri_adapt_reference,
                              "triangle_adaptive", adaptive_preview)
    
    # Create comparison plot
    print("\n--- Creating Error Comparison Plot ---")
    create_error_comparison_plot(
        quad_dofs_filtered, quad_errors,
        tri_dofs_filtered, tri_errors,
        tri_adapt_dofs_filtered, tri_adapt_errors
    )
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
