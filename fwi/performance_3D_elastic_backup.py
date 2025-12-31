import matplotlib.pyplot as plt
# latex rendering
plt.rc('text', usetex=True)
# mesh sizes
# dofs 8, 16, 24, 32, 40 (always 16 layers)
dofs = [70785, 274625, 611585, 1081665, 1684865, 2421185, 3290625, 4293185, 6697665, 9634625]

spefem_serial_time_dofs_8 = [5.09, 4.96, 4.94]
spefem_serial_time_dofs_16 = [13.28, 13.40, 13.03]
spefem_serial_time_dofs_24 = [48.42, 47.54, 46.89]
spefem_serial_time_dofs_32 = [85.29, 85.91, 85.72]
spefem_serial_time_dofs_40 = [134.39, 135.29, 135.29]
spefem_serial_time_dofs_48 = [196.84, 196.28, 193.95]
spefem_serial_time_dofs_56 = [271.21, 276.65, 273.12]
spefem_serial_time_dofs_64 = [359.63, 365.36, 355.39]
spefem_serial_time_dofs_80 = [558.71, 560.11, 562.22]
spefem_serial_time_dofs_96 = [799.80, 789.49, 826.71]

firedrake_serial_time_dofs_8 = [17.34, 17.34, 18.05]
firedrake_serial_time_dofs_16 = [63.29, 63.44,  63.27]
firedrake_serial_time_dofs_24 = [138.97,  139.38, 139.64]
firedrake_serial_time_dofs_32 = [244.99, 244.39, 245.14]
firedrake_serial_time_dofs_40 = [433.43, 433.27, 432.12]
firedrake_serial_time_dofs_48 = [617.46, 617.04, 619.45]
firedrake_serial_time_dofs_56 = [844.32, 846.02, 843.62]
firedrake_serial_time_dofs_64 = [1097.39, 1094.15, 1097.10]
firedrake_serial_time_dofs_80 = [1690.27,  1698.36, 1698.50]
firedrake_serial_time_dofs_96 = [2444.23, 2437.71, 2441.79]


# An array containing the average time for each mesh size
spefem_serial_time = [
    sum(spefem_serial_time_dofs_8)/3,
    sum(spefem_serial_time_dofs_16)/3,
    sum(spefem_serial_time_dofs_24)/3,
    sum(spefem_serial_time_dofs_32)/3,
    sum(spefem_serial_time_dofs_40)/3,
    sum(spefem_serial_time_dofs_48)/3,
    sum(spefem_serial_time_dofs_56)/3,
    sum(spefem_serial_time_dofs_64)/3,
    sum(spefem_serial_time_dofs_80)/3,
    sum(spefem_serial_time_dofs_96)/3]

firedrake_serial_time = [
    sum(firedrake_serial_time_dofs_8)/3,
    sum(firedrake_serial_time_dofs_16)/3,
    sum(firedrake_serial_time_dofs_24)/3,
    sum(firedrake_serial_time_dofs_32)/3,
    sum(firedrake_serial_time_dofs_40)/3,
    sum(firedrake_serial_time_dofs_48)/3,
    sum(firedrake_serial_time_dofs_56)/3,
    sum(firedrake_serial_time_dofs_64)/3,
    sum(firedrake_serial_time_dofs_80)/3,
    sum(firedrake_serial_time_dofs_96)/3]



# For 1000000 dofs, we test parallel performance
spefem_parallel_time_1cores = spefem_serial_time[-1]
spefem_parallel_time_2cores = [434.10, 433.57, 434.40]
spefem_parallel_time_4cores = [269.27, 268.54, 272.00]
spefem_parallel_time_8cores = [147.49, 149.06, 147.76]
spefem_parallel_time_12cores = [104.07, 102.65, 104.88]
spefem_parallel_time_16cores = [82.01, 80.56, 80.76]
spefem_parallel_time_24cores = [59.33, 60.05, 59.33]

firedrake_parallel_time_1cores = firedrake_serial_time[-1]
firedrake_parallel_time_2cores = [1269.39, 1282.24, 1284.99]
firedrake_parallel_time_4cores = [694.21, 696.64, 692.08]
firedrake_parallel_time_8cores = [392.38, 390.75, 391.12]
firedrake_parallel_time_12cores = [296.02, 295.70, 296.33]
firedrake_parallel_time_16cores = [271.57, 271.39, 270.82]
firedrake_parallel_time_24cores = [231.00, 230.79, 231.62]

# An array containing the average time for each number of cores
spefem_parallel_time = [
    spefem_parallel_time_1cores,
    sum(spefem_parallel_time_2cores)/3,
    sum(spefem_parallel_time_4cores)/3,
    sum(spefem_parallel_time_8cores)/3,
    sum(spefem_parallel_time_12cores)/3,
    sum(spefem_parallel_time_16cores)/3,
    sum(spefem_parallel_time_24cores)/3]

firedrake_parallel_time = [
    firedrake_parallel_time_1cores,
    sum(firedrake_parallel_time_2cores)/3,
    sum(firedrake_parallel_time_4cores)/3,
    sum(firedrake_parallel_time_8cores)/3,
    sum(firedrake_parallel_time_12cores)/3,
    sum(firedrake_parallel_time_16cores)/3,
    sum(firedrake_parallel_time_24cores)/3]



# Plot
plt.figure(figsize=(8, 6))
plt.loglog(dofs, spefem_serial_time, label=r"SPECFEM3D", marker="o", linestyle="--")
plt.loglog(dofs, firedrake_serial_time, label=r"Spyro", marker="s", linestyle="--")
plt.ylabel(r"Time (s)", fontsize=14)
plt.xlabel(r"Number of degrees of freedom", fontsize=14)
# Less intervals in y-axis
# plt.yticks([1, 10, 100, 500, 600, 700, 800, 1000])
plt.legend(fontsize=14)
plt.grid(True, which="both", ls="--")
plt.show()

# Plot
plt.figure(figsize=(8, 6))
plt.loglog([1, 2, 4, 8, 12, 16, 24], spefem_parallel_time, label=r"SPECFEM3D", marker="o", linestyle="--")
plt.loglog([1, 2, 4, 8, 12, 16, 24], firedrake_parallel_time, label=r"Spyro", marker="s", linestyle="--")
plt.xlabel(r"Number of processors", fontsize=14)
plt.ylabel(r"Time (s)", fontsize=14)
plt.legend(fontsize=14)
# Add more intervals in y-axis
plt.yticks([10, 100, 1000])
# Add more intervals in x-axis
plt.xticks([1, 2, 4, 8, 12, 16, 24, 32])
plt.grid(True, which="both", ls="--")
plt.show()