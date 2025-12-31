"""
Performance plotting script for elastic 2D simulations.
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
    """Manages performance data and calculations."""

    def __init__(self):
        # Mesh configuration
        self.mesh_config = {
            'dofs': [10201, 40401, 160801, 361201, 641601, 1002001],
            'mesh_sizes': [
                (25, 25), (50, 50), (100, 100),
                (150, 150), (200, 200), (250, 250)
            ]
        }

        # Serial performance data (3 measurements each)
        self.serial_data = {
            'specfem': PerformanceData([
                [0.35, 0.33, 0.35],      # 10201 dofs
                [1.33, 1.33, 1.33],      # 40401 dofs
                [5.34, 5.34, 5.36],      # 160801 dofs
                [12.46, 12.47, 12.41],   # 361201 dofs
                [22.42, 22.26, 22.18],   # 641601 dofs
                [33.88, 33.83, 34.2]     # 1002001 dofs
            ], "SPECFEM2D serial performance"),

            'firedrake': PerformanceData([
                [2.82, 2.78, 2.81],      # 10201 dofs
                [4.67, 4.64, 4.68],      # 40401 dofs
                [14.06, 14.15, 14.17],   # 160801 dofs
                [29.22, 30.81, 29.56],   # 361201 dofs
                [51.82, 51.96, 52.06],   # 641601 dofs
                [95.08, 92.98, 93.12]    # 1002001 dofs
            ], "Firedrake (Spyro) serial performance")
        }

        # Parallel performance data for 1M dofs (3 measurements each)
        self.parallel_data = {
            'cores': [1, 2, 4, 8, 12, 16, 24],
            'specfem': PerformanceData([
                None,                     # Will use serial time for 1 core
                [17.18, 17.18, 17.68],   # 2 cores
                [8.78, 8.81, 8.79],      # 4 cores
                [4.72, 4.81, 4.80],      # 8 cores
                [3.19, 3.19, 3.20],      # 12 cores
                [2.34, 2.94, 2.34],      # 16 cores
                [1.59, 1.58, 1.57]       # 24 cores
            ], "SPECFEM2D parallel performance"),

            'firedrake': PerformanceData([
                None,                     # Will use serial time for 1 core
                [42.19, 42.00, 42.05],   # 2 cores
                [22.93, 23.24, 22.66],   # 4 cores
                [15.57, 15.58, 15.60],   # 8 cores
                [13.30, 13.16, 13.20],   # 12 cores
                [10.68, 10.70, 10.72],   # 16 cores
                [9.53, 9.58, 9.57]       # 24 cores
            ], "Firedrake (Spyro) parallel performance")
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

        # Prepare parallel data with 1-core reference
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
        avg_serial_ratio = np.mean(list(serial_ratios.values())[2:])
        avg_parallel_ratio = np.mean(list(parallel_ratios.values())[2:])

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
    print("Starting performance plotting script...")

    # Initialize data manager and plotter
    data_manager = PerformanceDataManager()
    plotter = PerformancePlotter()

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
        title="Serial Performance Comparison",
        save_path="serial_performance.png"
    )
    print("Serial plot saved as 'serial_performance.png'")

    # Parallel performance plot
    print("Creating parallel performance plot...")
    plotter.create_plot(
        x_data=data_manager.parallel_data['cores'],
        y_data_dict=parallel_averages,
        x_label=r"Number of processors",
        y_label=r"Time (s)",
        title="Parallel Performance Comparison",
        save_path="parallel_performance.png"
    )
    print("Parallel plot saved as 'parallel_performance.png'")

    # Calculate and display speedup
    print("\nSpeedup Analysis (24 cores vs 1 core):")
    for solver, times in parallel_averages.items():
        speedup = times[0] / times[-1]  # 1 core time / 24 core time
        efficiency = speedup / 24 * 100  # Parallel efficiency
        print(f"{solver}: {speedup:.2f}x speedup, "
              f"{efficiency:.1f}% efficiency")

    # Performance ratio analysis
    print("\n" + "="*60)
    print("PERFORMANCE RATIO ANALYSIS")
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
          f"faster than Firedrake across all tests")

    print("\nScript completed successfully!")


if __name__ == "__main__":
    main()
