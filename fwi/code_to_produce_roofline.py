from firedrake import *
from firedrake.adjoint import *
from firedrake.__future__ import interpolate
import finat
import time
import numpy as np
import warnings
# Hide warnings
warnings.filterwarnings("ignore")

# Number of processors
M = 1
my_ensemble = Ensemble(COMM_WORLD, M)
num_sources = my_ensemble.ensemble_comm.size
source_number = my_ensemble.ensemble_comm.rank

mesh = UnitSquareMesh(100, 100, comm=my_ensemble.comm)

source_locations = np.linspace((0.3, 0.1), (0.7, 0.1), num_sources)
receiver_locations = np.linspace((0.2, 0.15), (0.8, 0.15), 100)
dt = 0.0001  # time step in seconds
total_steps = 100
frequency_peak = 5.0  # The dominant frequency of the Ricker wavelet in Hz.


V = FunctionSpace(mesh, "KMV", 1)
print("The number of degrees of freedom is", V.dim())

x, z = SpatialCoordinate(mesh)
c_true = Function(V).interpolate(1.75 + 0.25 * tanh(200 * (0.125 - sqrt((x - 0.5) ** 2 + (z - 0.5) ** 2))))


def ricker_wavelet(t, fs, amp=1.0):
    ts = 1.5
    t0 = t - ts * np.sqrt(6.0) / (np.pi * fs)
    return (amp * (1.0 - (1.0 / 2.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0)
            * np.exp((-1.0 / 4.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0))


source_mesh = VertexOnlyMesh(mesh, [source_locations[source_number]])
V_s = FunctionSpace(source_mesh, "DG", 0)
d_s = Function(V_s)
d_s.interpolate(1.0)
source_cofunction = assemble(d_s * TestFunction(V_s) * dx)

q_s = Cofunction(V.dual()).interpolate(source_cofunction)
quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree(), "KMV")


def wave_equation_solver(c, source_function, dt, V):
    u = TrialFunction(V)
    v = TestFunction(V)
    u_np1 = Function(V) # timestep n+1
    u_n = Function(V) # timestep n
    u_nm1 = Function(V) # timestep n-1
    time_term = (u - 2.0 * u_n + u_nm1) / Constant(dt**2) * v * dx(scheme=quad_rule)
    a = c * c * dot(grad(u_n), grad(v)) * dx(scheme=quad_rule)
    F = time_term + a
    lin_var = LinearVariationalProblem(lhs(F), rhs(F) + source_function, u_np1, constant_jacobian=True)
    solver_parameters = {"mat_type": "matfree", "ksp_type": "preonly", "pc_type": "jacobi"}
    solver = LinearVariationalSolver(lin_var, solver_parameters=solver_parameters)
    return solver, u_np1, u_n, u_nm1

receiver_mesh = VertexOnlyMesh(mesh, receiver_locations)
V_r = FunctionSpace(receiver_mesh, "DG", 0)

true_data_receivers = []

f = Cofunction(V.dual())  # Wave equation forcing term.
solver, u_np1, u_n, u_nm1 = wave_equation_solver(c_true, f, dt, V)
interpolate_receivers = interpolate(u_np1, V_r)
start = time.perf_counter()
for step in range(total_steps):
    f.assign(ricker_wavelet(step * dt, frequency_peak) * q_s)
    solver.solve()
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    true_data_receivers.append(assemble(interpolate_receivers))
end = time.perf_counter()
execution_time = end - start
print("The forward simulation took", execution_time, "seconds.")
# Number of degrees of freedom
N = V.dim()

# FLOPs per time step (approximate)
flops_per_step = 14 * N

# Total FLOPs
total_flops = flops_per_step * total_steps

# GFLOPS
gflops = total_flops / (execution_time * 1e9)

print(f"Total FLOPs: {total_flops}")
print(f"GFLOPS: {gflops}")

c_guess = Function(V).interpolate(1.5)

continue_annotation()
tape = get_working_tape()

f = Cofunction(V.dual())  # Wave equation forcing term.
solver, u_np1, u_n, u_nm1 = wave_equation_solver(c_guess, f, dt, V)
interpolate_receivers = interpolate(u_np1, V_r)
J_val = 0.0
start = time.perf_counter()
for step in tape.timestepper(iter(range(total_steps))):
    f.assign(ricker_wavelet(step * dt, frequency_peak) * q_s)
    solver.solve()
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    guess_receiver = assemble(interpolate_receivers)
    misfit = guess_receiver - true_data_receivers[step]
    J_val += 0.5 * assemble(inner(misfit, misfit) * dx)

end = time.perf_counter()
print("The annotation plus forward simulation took", end - start, "seconds.")

J_hat = EnsembleReducedFunctional(J_val, Control(c_guess), my_ensemble)
start = time.perf_counter()
J_hat(c_guess)
end = time.perf_counter()
print("The recompute simulation took", end - start, "seconds.")
start = time.perf_counter()
J_hat.derivative()
end = time.perf_counter()
print("The derivative simulation took", end - start, "seconds.")