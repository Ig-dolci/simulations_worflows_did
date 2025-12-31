"""
Performance plotting script for 3D elastic simulations.
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
        plt.show()
        return fig


class PerformanceDataManager:
    """Manages performance data and calculations for 3D elastic simulations."""

    def __init__(self):
        # Mesh configuration - 3D elastic with varying mesh sizes
        self.mesh_config = {
            'dofs': [70785, 274625, 611585, 1081665, 1684865, 2421185, 
                    3290625, 4293185, 6697665, 9634625],
            'mesh_descriptions': [
                '8x8x16', '16x16x16', '24x24x16', '32x32x16', '40x40x16',
                '48x48x16', '56x56x16', '64x64x16', '80x80x16', '96x96x16'
            ]
        }

        # Serial performance data (3 measurements each)
        self.serial_data = {
            'specfem': PerformanceData([
                [5.09, 4.96, 4.94],           # 70785 dofs
                [13.28, 13.40, 13.03],        # 274625 dofs
                [48.42, 47.54, 46.89],        # 611585 dofs
                [85.29, 85.91, 85.72],        # 1081665 dofs
                [134.39, 135.29, 135.29],     # 1684865 dofs
                [196.84, 196.28, 193.95],     # 2421185 dofs
                [271.21, 276.65, 273.12],     # 3290625 dofs
                [359.63, 365.36, 355.39],     # 4293185 dofs
                [558.71, 560.11, 562.22],     # 6697665 dofs
                [799.80, 789.49, 826.71]      # 9634625 dofs
            ], "SPECFEM3D elastic serial performance"),

            'firedrake': PerformanceData([
                [17.34, 17.34, 18.05],        # 70785 dofs
                [63.29, 63.44, 63.27],        # 274625 dofs
                [138.97, 139.38, 139.64],     # 611585 dofs
                [244.99, 244.39, 245.14],     # 1081665 dofs
                [433.43, 433.27, 432.12],     # 1684865 dofs
                [617.46, 617.04, 619.45],     # 2421185 dofs
                [844.32, 846.02, 843.62],     # 3290625 dofs
                [1097.39, 1094.15, 1097.10],  # 4293185 dofs
                [1690.27, 1698.36, 1698.50],  # 6697665 dofs
                [2444.23, 2437.71, 2441.79]   # 9634625 dofs
            ], "Firedrake (Spyro) elastic serial performance")
        }

        # Parallel performance data for largest mesh (9634625 dofs)
        self.parallel_data = {
            'cores': [1, 2, 4, 8, 12, 16, 24],
            'specfem': PerformanceData([
                None,                          # Will use serial time for 1 core
                [434.10, 433.57, 434.40],     # 2 cores
                [269.27, 268.54, 272.00],     # 4 cores
                [147.49, 149.06, 147.76],     # 8 cores
                [104.07, 102.65, 104.88],     # 12 cores
                [82.01, 80.56, 80.76],        # 16 cores
                [59.33, 60.05, 59.33]         # 24 cores
            ], "SPECFEM3D elastic parallel performance"),

            'firedrake': PerformanceData([
                None,                          # Will use serial time for 1 core
                [1269.39, 1282.24, 1284.99],  # 2 cores
                [694.21, 696.64, 692.08],     # 4 cores
                [392.38, 390.75, 391.12],     # 8 cores
                [296.02, 295.70, 296.33],     # 12 cores
                [271.57, 271.39, 270.82],     # 16 cores
                [231.00, 230.79, 231.62]      # 24 cores
            ], "Firedrake (Spyro) elastic parallel performance")
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
    print("Starting 3D elastic performance plotting script...")
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
        save_path="elastic_3d_serial_performance.eps"
    )
    print("Serial plot saved as 'elastic_3d_serial_performance.eps'")

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
        save_path="elastic_3d_parallel_performance.eps",
        custom_ticks={
            'yticks': [10, 100, 1000],
            'xticks': [1, 2, 4, 8, 12, 16, 24, 32]
        }
    )
    print("Parallel plot saved as 'elastic_3d_parallel_performance.eps'")

    # Calculate and display speedup
    print("\nSpeedup Analysis (24 cores vs 1 core):")
    for solver, times in parallel_averages.items():
        speedup = times[0] / times[-1]  # 1 core time / 24 core time
        efficiency = speedup / 24 * 100  # Parallel efficiency
        print(f"{solver}: {speedup:.2f}x speedup, "
              f"{efficiency:.1f}% efficiency")

    # Performance ratio analysis
    print("\n" + "="*60)
    print("3D ELASTIC PERFORMANCE RATIO ANALYSIS")
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
          f"faster than Firedrake across all 3D elastic tests")

    print("\nScript completed successfully!")


if __name__ == "__main__":
    main()