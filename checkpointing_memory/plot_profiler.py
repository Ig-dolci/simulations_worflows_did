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


def compute_max_diff_no_annotation(data_no_revolve, data_no_annotation):
    max_diff = abs(max(data_no_revolve) - max(data_no_annotation))
    return max_diff


if __name__ == '__main__':
    data_no_revolve_int = parse_memory_profiler('no_revolve_interpolate')
    data_revolve_s10_int = parse_memory_profiler('revolve_s10_interpolate')
    data_revolve_s100_int = parse_memory_profiler('revolve_s100_interpolate')
    data_revolve_s200_int = parse_memory_profiler('revolve_s200_interpolate')
    data_revolve_s400_int = parse_memory_profiler('revolve_s400_interpolate')
    data_no_annotation_int = parse_memory_profiler('no_annotation_interpolate')

    data_no_revolve_assigns = parse_memory_profiler('no_revolve_assign')
    data_revolve_s10_assigns = parse_memory_profiler('revolve_s10_assign')
    data_revolve_s100_assigns = parse_memory_profiler('revolve_s100_assign')
    data_revolve_s200_assigns = parse_memory_profiler('revolve_s200_assign')
    data_revolve_s400_assigns = parse_memory_profiler('revolve_s400_assign')
    data_no_annotation_assigns = parse_memory_profiler('no_annotation_assign')

    data_no_revolve_burgers = parse_memory_profiler('no_revolve_burger')
    data_revolve_s10_burgers = parse_memory_profiler('revolve_s10_burger')
    data_revolve_s100_burgers = parse_memory_profiler('revolve_s100_burger')
    data_revolve_s200_burgers = parse_memory_profiler('revolve_s200_burger')
    data_revolve_s400_burgers = parse_memory_profiler('revolve_s400_burger')
    data_no_annotation_burgers = parse_memory_profiler('no_annotation_burger')
    diff_int = []
    diff_int.append(compute_max_diff_no_annotation(data_revolve_s10_int, data_no_annotation_int))
    diff_int.append(compute_max_diff_no_annotation(data_revolve_s100_int, data_no_annotation_int))
    diff_int.append(compute_max_diff_no_annotation(data_revolve_s200_int, data_no_annotation_int))
    diff_int.append(compute_max_diff_no_annotation(data_revolve_s400_int, data_no_annotation_int))
    diff_int.append(compute_max_diff_no_annotation(data_no_revolve_int, data_no_annotation_int))

    diff_assigns = []
    diff_assigns.append(compute_max_diff_no_annotation(data_revolve_s10_assigns, data_no_annotation_assigns))
    diff_assigns.append(compute_max_diff_no_annotation(data_revolve_s100_assigns, data_no_annotation_assigns))
    diff_assigns.append(compute_max_diff_no_annotation(data_revolve_s200_assigns, data_no_annotation_assigns))
    diff_assigns.append(compute_max_diff_no_annotation(data_revolve_s400_assigns, data_no_annotation_assigns))
    diff_assigns.append(compute_max_diff_no_annotation(data_no_revolve_assigns, data_no_annotation_assigns))

    diff_burgers = []
    diff_burgers.append(compute_max_diff_no_annotation(data_revolve_s10_burgers, data_no_annotation_burgers))
    diff_burgers.append(compute_max_diff_no_annotation(data_revolve_s100_burgers, data_no_annotation_burgers))
    diff_burgers.append(compute_max_diff_no_annotation(data_revolve_s200_burgers, data_no_annotation_burgers))
    diff_burgers.append(compute_max_diff_no_annotation(data_revolve_s400_burgers, data_no_annotation_burgers))
    diff_burgers.append(compute_max_diff_no_annotation(data_no_revolve_burgers, data_no_annotation_burgers))


    normalised_diff_int = [d / diff_int[-1] for d in diff_int]
    normalised_diff_assigns = [d / diff_assigns[-1] for d in diff_assigns]
    normalised_diff_burgers = [d / diff_burgers[-1] for d in diff_burgers]
    percentage = [0.01, 0.1, 0.2, 0.4, 1.0]
    print("Normalised diff int: ", normalised_diff_int)
    print("Normalised diff assigns: ", normalised_diff_assigns)
    print("Normalised diff burgers: ", normalised_diff_burgers)
    plt.plot(percentage[:-1], normalised_diff_int[:-1], 'bs', label='Interpolate')
    plt.plot(percentage[:-1], normalised_diff_assigns[:-1], 'g^', label='Assign')
    plt.plot(percentage[:-1], normalised_diff_burgers[:-1], 'yo', label='Burgers')
    # plot y = x
    plt.plot(percentage, percentage, 'k-.', label='y=x')
    plt.plot(percentage, [2 * p for p in percentage], 'r--', label='y=2x')
    plt.xlabel(r'$n_r = n_{chk}/n_T$', fontsize=16)
    plt.ylabel(r'$M$', fontsize=16)
    plt.grid(which='both', linestyle='--')
    plt.xticks(fontsize=12)
    plt.legend(fontsize=12)
    # decreaser the interval of x-axis
    plt.xticks([0.01, 0.1, 0.2, 0.4])
    plt.show()