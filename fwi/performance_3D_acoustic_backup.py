import matplotlib.pyplot as plt
# latex rendering
plt.rc('text', usetex=True)
# mesh sizes
# dofs 8, 16, 24, 32, 40 (always 16 layers)
dofs = [70785, 274625, 611585, 1081665, 1684865, 2421185, 3290625, 4293185, 6697665, 9634625]
# This is used to pin the python process. With this I can reach the maximal memory bandwith
# of this machine.
# mpiexec -n 32 --map-by L3CACHE --bind-to CORE --rank-by SPAN python cpu.py
# Read this https://petsc.org/release/manual/streams/
# This is to have the best petsc memory bandwith of the machine.
# make mpistream N=640000000 MPI_BINDING="--map-by L3CACHE --bind-to CORE --rank-by SPAN" NPMAX=32

spefem_serial_time_dofs_8 = [1.44, 1.45, 1.44]
spefem_serial_time_dofs_16 = [3.23, 3.26, 3.25]
spefem_serial_time_dofs_24 = [13.93, 13.85, 13.83]
spefem_serial_time_dofs_32 = [25.07, 25.00, 24.83]
spefem_serial_time_dofs_40 = [40.69,  40.79,  40.22]
spefem_serial_time_dofs_48 = [59.78, 59.54, 59.80]
spefem_serial_time_dofs_56 = [81.09, 80.97,  81.27]
spefem_serial_time_dofs_64 = [111.31, 109.34, 110.20]
spefem_serial_time_dofs_80 = [177.0, 175.44, 177.11]
spefem_serial_time_dofs_96 = [258.924,  257.61, 258.00]

firedrake_serial_time_dofs_8 = [8.540, 8.40, 8.50]
firedrake_serial_time_dofs_16 = [27.20, 27.04, 27.18]
firedrake_serial_time_dofs_24 = [58.75, 59.04, 59.19]
firedrake_serial_time_dofs_32 = [103.95, 104.24, 103.73]
firedrake_serial_time_dofs_40 = [161.30, 161.12, 160.67]
firedrake_serial_time_dofs_48 = [231.72, 232.01, 232.08]
firedrake_serial_time_dofs_56 = [330.32, 323.50, 326.00]
firedrake_serial_time_dofs_64 = [430.25, 429.50, 430.00]
firedrake_serial_time_dofs_80 = [664.76,  666.12, 665.00]
firedrake_serial_time_dofs_96 = [955.60,  956.00,  955.50]


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
spefem_parallel_time_2cores = [145.05, 145.31, 145.25]
spefem_parallel_time_4cores = [93.56, 91.22, 92.39]
spefem_parallel_time_8cores = [78.71, 78.54, 75.26]
spefem_parallel_time_12cores = [50.67, 50.56, 51.15]
spefem_parallel_time_16cores = [39.52, 39.72, 39.99]
spefem_parallel_time_24cores = [30.22,  30.19, 30.00]

firedrake_parallel_time_1cores = firedrake_serial_time[-1]
firedrake_parallel_time_2cores = [496.49,  499.20, 499.93]
firedrake_parallel_time_4cores = [248.60,  247.5,  248.15]
firedrake_parallel_time_8cores = [147.49, 146.95, 147.01]
firedrake_parallel_time_12cores = [113.20, 113.32, 113.15]
firedrake_parallel_time_16cores = [99.08, 99.24, 99.25]
firedrake_parallel_time_24cores = [81.55, 81.69, 81.60]

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