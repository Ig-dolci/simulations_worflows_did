"""
Performance plotting script for scalar 2D simulations.
Compares SPECFEM2D and Spyro (Firedrake) performance.
"""

import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union

# Configure matplotlib - disable LaTeX for compatibility
# plt.rc('text', usetex=True)


@dataclass
class PlotStyle:
    """Configuration for plot styling."""
    marker: str
    linestyle: str
    color: Optional[str] = None


@dataclass
class PerformanceData:
    """Container for performance measurement data."""
    measurements: List[List[float]]
    description: Optional[str] = None

    @property
    def averages(self) -> List[float]:
        """Calculate average times from repeated measurements."""
        return [np.mean(times) for times in self.measurements]


class PerformancePlotter:
    """Enhanced performance plotting class with better organization."""

    def __init__(self, figsize: Tuple[int, int] = (10, 7), fontsize: int = 14):
        self.figsize = figsize
        self.fontsize = fontsize
        self.styles = {
            'SPECFEM2D': PlotStyle(
                marker="o", linestyle="--", color='#1f77b4'
            ),
            'Spyro': PlotStyle(
                marker="s", linestyle="--", color='#ff7f0e'
            )
        }

    def create_plot(self, x_data: List[Union[int, float]],
                    y_data_dict: Dict[str, List[float]],
                    x_label: str, y_label: str,
                    title: Optional[str] = None,
                    log_scale: bool = True,
                    save_path: Optional[str] = None) -> plt.Figure:
        """Create a performance plot with enhanced styling."""
        fig, ax = plt.subplots(figsize=self.figsize)

        for label, y_data in y_data_dict.items():
            style = self.styles.get(
                label, PlotStyle(marker="o", linestyle="-")
            )

            plot_kwargs = {
                'label': label,
                'marker': style.marker,
                'linestyle': style.linestyle,
                'color': style.color,
                'markersize': 8,
                'linewidth': 2
            }

            if log_scale:
                ax.loglog(x_data, y_data, **plot_kwargs)
            else:
                ax.plot(x_data, y_data, **plot_kwargs)

        ax.set_xlabel(x_label, fontsize=self.fontsize)
        ax.set_ylabel(y_label, fontsize=self.fontsize)

        if title:
            ax.set_title(title, fontsize=self.fontsize + 2)

        ax.legend(fontsize=self.fontsize)
        ax.grid(True, which="both", ls="--", alpha=0.7)

        # Improve tick label size
        ax.tick_params(
            axis='both', which='major', labelsize=self.fontsize - 2
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class PerformanceDataManager:
    """Manages performance data and calculations for scalar simulations."""

    def __init__(self):
        # Mesh configuration
        self.mesh_config = {
            'dofs': [10201, 40401, 160801, 361201, 641601, 1002001, 1442401, 1962801],
            'mesh_sizes': [
                (25, 25), (50, 50), (100, 100), (150, 150),
                (200, 200), (250, 250), (300, 300), (350, 350)
            ]
        }

        # Serial performance data (3 measurements each)
        self.serial_data = {
            'specfem': PerformanceData([
                [0.11, 0.11, 0.11],           # 10201 dofs
                [0.45, 0.46, 0.45],           # 40401 dofs
                [1.83, 1.82, 1.82],           # 160801 dofs
                [4.10, 4.11, 4.13],           # 361201 dofs
                [7.40, 7.49, 7.40],           # 641601 dofs
                [11.75, 11.67, 11.76],        # 1002001 dofs
                [16.71, 16.68, 16.70],        # 1442401 dofs
                [22.75, 22.74, 22.76]         # 1962801 dofs
            ], "SPECFEM2D scalar serial performance"),

            'firedrake': PerformanceData([
                [2.5, 2.52, 2.49],            # 10201 dofs
                [3.44, 3.47, 3.48],           # 40401 dofs
                [7.74, 7.76, 7.69],           # 160801 dofs
                [15.54, 15.44, 15.52],        # 361201 dofs
                [29.36, 26.57, 26.59],        # 641601 dofs
                [40.68, 40.92, 41.03],        # 1002001 dofs
                [58.75, 58.80, 59.40],        # 1442401 dofs
                [80.02, 79.50, 79.60]         # 1962801 dofs
            ], "Firedrake (Spyro) scalar serial performance")
        }

        # Parallel performance data for largest dofs (3 measurements each)
        self.parallel_data = {
            'cores': [1, 2, 4, 8, 12, 16, 24],
            'specfem': PerformanceData([
                None,                         # Will use serial time for 1 core
                [11.73, 11.73, 11.73],       # 2 cores
                [6.14, 6.07, 6.10],          # 4 cores
                [3.37, 3.33, 3.35],          # 8 cores
                [1.08, 1.07, 1.08],          # 12 cores
                [0.81, 0.82, 0.81],          # 16 cores
                [0.53, 0.54, 0.53]           # 24 cores
            ], "SPECFEM2D scalar parallel performance"),

            'firedrake': PerformanceData([
                None,                         # Will use serial time for 1 core
                [42.42, 42.26, 42.30],       # 2 cores
                [22.12, 22.32, 22.20],       # 4 cores
                [14.15, 14.15, 14.18],       # 8 cores
                [12.50, 12.85, 12.60],       # 12 cores
                [11.58, 11.60, 11.55],       # 16 cores
                [10.20, 10.30, 10.40]        # 24 cores
            ], "Firedrake (Spyro) scalar parallel performance")
        }

    def get_serial_averages(self) -> Dict[str, List[float]]:
        """Get averaged serial performance data."""
        return {
            'SPECFEM2D': self.serial_data['specfem'].averages,
            'Spyro': self.serial_data['firedrake'].averages
        }

    def get_parallel_averages(self) -> Dict[str, List[float]]:
        """Get averaged parallel performance data."""
        # Get serial averages for 1-core reference
        serial_avgs = self.get_serial_averages()

        # Prepare parallel data with 1-core reference (use largest DOF)
        specfem_parallel = [serial_avgs['SPECFEM2D'][-1]]
        firedrake_parallel = [serial_avgs['Spyro'][-1]]

        # Add parallel measurements (skip first None entry)
        parallel_measurements = self.parallel_data['specfem'].measurements[1:]
        specfem_parallel.extend([np.mean(times) for times in
                                parallel_measurements])

        parallel_firedrake = self.parallel_data['firedrake'].measurements[1:]
        firedrake_parallel.extend([np.mean(times) for times in
                                  parallel_firedrake])

        return {
            'SPECFEM2D': specfem_parallel,
            'Spyro': firedrake_parallel
        }

    def calculate_serial_performance_ratios(self) -> Dict[str, float]:
        """Calculate how many times faster SPECFEM is vs Firedrake (serial)."""
        serial_avgs = self.get_serial_averages()
        specfem_times = serial_avgs['SPECFEM2D']
        firedrake_times = serial_avgs['Spyro']

        ratios = {}
        for i, dofs in enumerate(self.mesh_config['dofs']):
            ratio = firedrake_times[i] / specfem_times[i]
            ratios[f"{dofs}_dofs"] = ratio

        return ratios

    def calculate_parallel_performance_ratios(self) -> Dict[str, float]:
        """Calculate how many times faster SPECFEM is vs Firedrake
        (parallel)."""
        parallel_avgs = self.get_parallel_averages()
        specfem_times = parallel_avgs['SPECFEM2D']
        firedrake_times = parallel_avgs['Spyro']

        ratios = {}
        for i, cores in enumerate(self.parallel_data['cores']):
            ratio = firedrake_times[i] / specfem_times[i]
            ratios[f"{cores}_cores"] = ratio

        return ratios

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary including ratios."""
        serial_ratios = self.calculate_serial_performance_ratios()
        parallel_ratios = self.calculate_parallel_performance_ratios()

        # Calculate average ratios
        avg_serial_ratio = np.mean(list(serial_ratios.values()))
        avg_parallel_ratio = np.mean(list(parallel_ratios.values()))

        return {
            'serial_ratios': serial_ratios,
            'parallel_ratios': parallel_ratios,
            'avg_serial_ratio': avg_serial_ratio,
            'avg_parallel_ratio': avg_parallel_ratio,
            'overall_avg_ratio': np.mean([avg_serial_ratio,
                                         avg_parallel_ratio])
        }


def main():
    """Main function to generate performance plots and analysis."""
    print("Starting scalar 2D performance plotting script...")
    print("Script is running...")

    # Initialize data manager and plotter
    data_manager = PerformanceDataManager()
    plotter = PerformancePlotter()
    print("Initialized data manager and plotter...")

    # Get averaged data
    serial_averages = data_manager.get_serial_averages()
    parallel_averages = data_manager.get_parallel_averages()

    print("Data loaded successfully.")
    print(f"Serial data points: {len(serial_averages['SPECFEM2D'])}")
    print(f"Parallel data points: {len(parallel_averages['SPECFEM2D'])}")

    # Serial performance plot (skip first 2 points as in original)
    serial_plot_data = {
        'SPECFEM2D': serial_averages['SPECFEM2D'][2:],
        'Spyro': serial_averages['Spyro'][2:]
    }

    print("Creating serial performance plot...")
    plotter.create_plot(
        x_data=data_manager.mesh_config['dofs'][2:],
        y_data_dict=serial_plot_data,
        x_label=r"Number of degrees of freedom",
        y_label=r"Time (s)",
        title="Scalar Serial Performance Comparison",
        save_path="scalar_serial_performance.png"
    )
    print("Serial plot saved as 'scalar_serial_performance.png'")

    # Parallel performance plot (limit to first 4 points as in original)
    parallel_plot_data = {
        'SPECFEM2D': parallel_averages['SPECFEM2D'][:4],
        'Spyro': parallel_averages['Spyro'][:4]
    }

    print("Creating parallel performance plot...")
    plotter.create_plot(
        x_data=data_manager.parallel_data['cores'][:4],
        y_data_dict=parallel_plot_data,
        x_label=r"Number of processors",
        y_label=r"Time (s)",
        title="Scalar Parallel Performance Comparison",
        save_path="scalar_parallel_performance.png"
    )
    print("Parallel plot saved as 'scalar_parallel_performance.png'")

    # Calculate and display speedup
    print("\nSpeedup Analysis (8 cores vs 1 core):")
    for solver, times in parallel_averages.items():
        speedup = times[0] / times[3]  # 1 core time / 8 core time
        efficiency = speedup / 8 * 100  # Parallel efficiency
        print(f"{solver}: {speedup:.2f}x speedup, "
              f"{efficiency:.1f}% efficiency")

    # Performance ratio analysis
    print("\n" + "="*60)
    print("SCALAR PERFORMANCE RATIO ANALYSIS")
    print("="*60)

    summary = data_manager.get_performance_summary()

    print("\nSerial Performance Ratios (Firedrake/SPECFEM):")
    print("-"*50)
    for config, ratio in summary['serial_ratios'].items():
        dofs = config.replace('_dofs', '')
        print(f"  {dofs:>8} DOFs: {ratio:.2f}x")

    print(f"\nAverage serial ratio: {summary['avg_serial_ratio']:.2f}x")
    print(f"SPECFEM is {summary['avg_serial_ratio']:.2f}x faster than "
          f"Firedrake (serial)")

    print("\nParallel Performance Ratios (Firedrake/SPECFEM):")
    print("-"*50)
    for config, ratio in summary['parallel_ratios'].items():
        cores = config.replace('_cores', '')
        print(f"  {cores:>2} cores: {ratio:.2f}x")

    print(f"\nAverage parallel ratio: {summary['avg_parallel_ratio']:.2f}x")
    print(f"SPECFEM is {summary['avg_parallel_ratio']:.2f}x faster than "
          f"Firedrake (parallel)")

    print("\nOverall Summary:")
    print("-"*50)
    print(f"Overall average ratio: {summary['overall_avg_ratio']:.2f}x")
    print(f"SPECFEM is approximately {summary['overall_avg_ratio']:.2f}x "
          f"faster than Firedrake across all scalar tests")

    print("\nScript completed successfully!")


if __name__ == "__main__":
    main()
