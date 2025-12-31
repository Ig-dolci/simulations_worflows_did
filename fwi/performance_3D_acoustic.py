"""
Performance plotting script for 3D acoustic simulations.
Compares SPECFEM3D and Spyro (Firedrake) performance.
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
            'SPECFEM3D': PlotStyle(
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
                    save_path: Optional[str] = None,
                    custom_ticks: Optional[Dict] = None) -> plt.Figure:
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

        # Apply custom ticks if provided
        if custom_ticks:
            if 'yticks' in custom_ticks:
                ax.set_yticks(custom_ticks['yticks'])
            if 'xticks' in custom_ticks:
                ax.set_xticks(custom_ticks['xticks'])

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
    """Manages performance data and calculations for 3D acoustic simulations."""

    def __init__(self):
        # Mesh configuration
        self.mesh_config = {
            'dofs': [
                # 70785, 274625
                611585, 1081665, 1684865, 2421185,
                     3290625, 4293185, 6697665, 9634625],
            'mesh_sizes': [
                (8, 16), (16, 16), (24, 16), (32, 16), (40, 16),
                (48, 16), (56, 16), (64, 16), (80, 16), (96, 16)
            ]
        }

        # Serial performance data (3 measurements each)
        self.serial_data = {
            'specfem': PerformanceData([
                # [1.44, 1.45, 1.44],               # 70785 dofs
                # [3.23, 3.26, 3.25],               # 274625 dofs
                [13.93, 13.85, 13.83],            # 611585 dofs
                [25.07, 25.00, 24.83],            # 1081665 dofs
                [40.69, 40.79, 40.22],            # 1684865 dofs
                [59.78, 59.54, 59.80],            # 2421185 dofs
                [81.09, 80.97, 81.27],            # 3290625 dofs
                [111.31, 109.34, 110.20],         # 4293185 dofs
                [177.0, 175.44, 177.11],          # 6697665 dofs
                [258.924, 257.61, 258.00]         # 9634625 dofs
            ], "SPECFEM3D acoustic serial performance"),

            'firedrake': PerformanceData([
                # [8.540, 8.40, 8.50],              # 70785 dofs
                # [27.20, 27.04, 27.18],            # 274625 dofs
                [58.75, 59.04, 59.19],            # 611585 dofs
                [103.95, 104.24, 103.73],         # 1081665 dofs
                [161.30, 161.12, 160.67],         # 1684865 dofs
                [231.72, 232.01, 232.08],         # 2421185 dofs
                [330.32, 323.50, 326.00],         # 3290625 dofs
                [430.25, 429.50, 430.00],         # 4293185 dofs
                [664.76, 666.12, 665.00],         # 6697665 dofs
                [955.60, 956.00, 955.50]          # 9634625 dofs
            ], "Firedrake (Spyro) acoustic serial performance")
        }

        # Parallel performance data for largest dofs (3 measurements each)
        self.parallel_data = {
            'cores': [1, 2, 4, 8, 12, 16, 24],
            'specfem': PerformanceData([
                None,                             # Will use serial time for 1 core
                [145.05, 145.31, 145.25],        # 2 cores
                [93.56, 91.22, 92.39],           # 4 cores
                [78.71, 78.54, 75.26],           # 8 cores
                [50.67, 50.56, 51.15],           # 12 cores
                [39.52, 39.72, 39.99],           # 16 cores
                [30.22, 30.19, 30.00]            # 24 cores
            ], "SPECFEM3D acoustic parallel performance"),

            'firedrake': PerformanceData([
                None,                             # Will use serial time for 1 core
                [496.49, 499.20, 499.93],        # 2 cores
                [248.60, 247.5, 248.15],         # 4 cores
                [147.49, 146.95, 147.01],        # 8 cores
                [113.20, 113.32, 113.15],        # 12 cores
                [99.08, 99.24, 99.25],           # 16 cores
                [81.55, 81.69, 81.60]            # 24 cores
            ], "Firedrake (Spyro) acoustic parallel performance")
        }

    def get_serial_averages(self) -> Dict[str, List[float]]:
        """Get averaged serial performance data."""
        return {
            'SPECFEM3D': self.serial_data['specfem'].averages,
            'Spyro': self.serial_data['firedrake'].averages
        }

    def get_parallel_averages(self) -> Dict[str, List[float]]:
        """Get averaged parallel performance data."""
        # Get serial averages for 1-core reference
        serial_avgs = self.get_serial_averages()

        # Prepare parallel data with 1-core reference (use largest DOF)
        specfem_parallel = [serial_avgs['SPECFEM3D'][-1]]
        firedrake_parallel = [serial_avgs['Spyro'][-1]]

        # Add parallel measurements (skip first None entry)
        parallel_measurements = self.parallel_data['specfem'].measurements[1:]
        specfem_parallel.extend([np.mean(times) for times in
                                parallel_measurements])

        parallel_firedrake = self.parallel_data['firedrake'].measurements[1:]
        firedrake_parallel.extend([np.mean(times) for times in
                                  parallel_firedrake])

        return {
            'SPECFEM3D': specfem_parallel,
            'Spyro': firedrake_parallel
        }

    def calculate_serial_performance_ratios(self) -> Dict[str, float]:
        """Calculate how many times faster SPECFEM is vs Firedrake (serial)."""
        serial_avgs = self.get_serial_averages()
        specfem_times = serial_avgs['SPECFEM3D']
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
        specfem_times = parallel_avgs['SPECFEM3D']
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
    print("Starting 3D acoustic performance plotting script...")
    print("Script is running...")

    # Initialize data manager and plotter
    data_manager = PerformanceDataManager()
    plotter = PerformancePlotter()
    print("Initialized data manager and plotter...")

    # Get averaged data
    serial_averages = data_manager.get_serial_averages()
    parallel_averages = data_manager.get_parallel_averages()

    print("Data loaded successfully.")
    print(f"Serial data points: {len(serial_averages['SPECFEM3D'])}")
    print(f"Parallel data points: {len(parallel_averages['SPECFEM3D'])}")

    # Serial performance plot (use all points)
    serial_plot_data = {
        'SPECFEM3D': serial_averages['SPECFEM3D'],
        'Spyro': serial_averages['Spyro']
    }

    print("Creating serial performance plot...")
    plotter.create_plot(
        x_data=data_manager.mesh_config['dofs'],
        y_data_dict=serial_plot_data,
        x_label=r"Number of degrees of freedom",
        y_label=r"Time (s)",
        title="3D Acoustic Serial Performance Comparison",
        save_path="acoustic_3d_serial_performance.png"
    )
    print("Serial plot saved as 'acoustic_3d_serial_performance.png'")

    # Parallel performance plot with custom ticks
    parallel_plot_data = {
        'SPECFEM3D': parallel_averages['SPECFEM3D'],
        'Spyro': parallel_averages['Spyro']
    }

    print("Creating parallel performance plot...")
    plotter.create_plot(
        x_data=data_manager.parallel_data['cores'],
        y_data_dict=parallel_plot_data,
        x_label=r"Number of processors",
        y_label=r"Time (s)",
        title="3D Acoustic Parallel Performance Comparison",
        save_path="acoustic_3d_parallel_performance.png",
        custom_ticks={
            'yticks': [10, 100, 1000],
            'xticks': [1, 2, 4, 8, 12, 16, 24, 32]
        }
    )
    print("Parallel plot saved as 'acoustic_3d_parallel_performance.png'")

    # Calculate and display speedup
    print("\nSpeedup Analysis (24 cores vs 1 core):")
    for solver, times in parallel_averages.items():
        speedup = times[0] / times[-1]  # 1 core time / 24 core time
        efficiency = speedup / 24 * 100  # Parallel efficiency
        print(f"{solver}: {speedup:.2f}x speedup, "
              f"{efficiency:.1f}% efficiency")

    # Performance ratio analysis
    print("\n" + "="*60)
    print("3D ACOUSTIC PERFORMANCE RATIO ANALYSIS")
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
          f"faster than Firedrake across all 3D acoustic tests")

    print("\nScript completed successfully!")


if __name__ == "__main__":
    main()
