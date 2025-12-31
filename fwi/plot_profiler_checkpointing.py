import re
import matplotlib.pyplot as plt
# use latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def parse_memory_profiler(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    memory = []
    for line in lines:
        if line.startswith('MEM'):
            memory.append(float(re.findall(r'\d+\.\d+', line)[0]))
    return memory

def plot_memory_profiler(filename):
    memory = parse_memory_profiler(filename)
    plt.plot(memory)
    plt.show()


def compute_max_diff_no_annotation(data_no_revolve, data_nocheckpoint):
    max_diff = abs(max(data_no_revolve) - max(data_nocheckpoint))
    return max_diff


if __name__ == '__main__':
    data_single = parse_memory_profiler('comp_performance/test_checkpoint_single')
    data_revolve = parse_memory_profiler('comp_performance/test_checkpoint_revolve')
    data_mixed = parse_memory_profiler('comp_performance/test_checkpoint_mixed')
    data_base_case = parse_memory_profiler('comp_performance/test_base_case')
    data_nocheckpoint = parse_memory_profiler('comp_performance/test_nocheckpoint')
    # data_noannotation = parse_memory_profiler('comp_performance/test_annotation')
    # data_only_taping = parse_memory_profiler('comp_performance/test_only_taping')

    # base_single = compute_max_diff_no_annotation(data_nocheckpoint, data_single)
    # revolve = compute_max_diff_no_annotation(data_nocheckpoint, data_revolve)
    # mixed = compute_max_diff_no_annotation(data_nocheckpoint, data_mixed)
    # print(base_single, revolve, mixed)

    normalised_base_single = (max(data_single)) / max(data_nocheckpoint)
    normalised_revolve = (max(data_revolve)) / max(data_nocheckpoint)
    normalised_mixed = (max(data_mixed)) / max(data_nocheckpoint)
    normalised_memory = [1.0, normalised_base_single, normalised_revolve, normalised_mixed]

    # Plot bar chart with the string labels in the x-axis
    # and normalised values in the y-axis
    # Create the figure and axes
    categories = ['No Checkpoint', 'Single Shedule', 'Revolve', 'Mixed Schedule']
    plt.figure(figsize=(8, 6))
    bars = plt.bar(
        categories,
        normalised_memory,
        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],  # Custom colors for bars
        edgecolor="black",  # Add black edges for contrast
        linewidth=1.5
    )

    # Add annotations on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            height + 0.02,  # Position above the bar
            f"{height:.2f}",  # Show value with 1 decimal
            ha='center', 
            va='bottom', 
            fontsize=12, 
            color='black'
        )

    # Customize axis labels and ticks
    plt.ylabel(r'Normalised memory usage', fontsize=18, labelpad=10)
    plt.xlabel(r'Checkpointing strategies', fontsize=18, labelpad=10)
    plt.yticks([i / 10 for i in range(0, 11)], fontsize=14)
    plt.xticks(fontsize=14)

    # Add gridlines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Get the maximum time from the dat files
    time_single = 149.01
    time_revolve = 304.94
    time_mixed = 211.9
    time_nocheckpoint = 144.12

    single = (time_single) / (time_nocheckpoint)
    revolve = (time_revolve) / (time_nocheckpoint)
    mixed = (time_mixed) / (time_nocheckpoint)
    normalised_time = [1.0, single, revolve, mixed]

    # In expense non-project code: 
    # Plot bar chart with the string labels in the x-axis
    # and normalised values in the y-axis

    # Create the figure and axes
    # categories = ['Single', 'Revolve', 'Mixed']
    plt.figure(figsize=(8, 6))
    bars = plt.bar(
        categories,
        normalised_time,
        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],  # Custom colors for bars
        edgecolor="black",  # Add black edges for contrast
        linewidth=1.5
    )

    # Add annotations on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            height + 0.02,  # Position above the bar
            f"{height:.2f}",  # Show value with 1 decimal
            ha='center', 
            va='bottom', 
            fontsize=12, 
            color='black'
        )

    # Customize axis labels and ticks
    plt.ylabel(r'Normalised time execution', fontsize=18, labelpad=10)
    plt.xlabel(r'Checkpointing strategies', fontsize=18, labelpad=10)
    plt.yticks([i / 2 for i in range(0, 6)], fontsize=14)
    plt.xticks(fontsize=14)

    # Add gridlines for better readability until the maximum value
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()