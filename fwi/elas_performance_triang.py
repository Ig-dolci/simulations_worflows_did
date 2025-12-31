from tools import model_interpolate
from firedrake import *
import finat
import FIAT
from firedrake.__future__ import interpolate
import numpy as np
import time
from mpi4py import MPI
import scipy.ndimage
# read a hdf5 file
M = 1
my_ensemble = Ensemble(MPI.COMM_WORLD, M)
num_sources = my_ensemble.ensemble_comm.size
source_number = my_ensemble.ensemble_comm.rank
mesh = RectangleMesh(380, 120, 200.0, 80.0, comm=my_ensemble.comm)
dt = 0.01 # time step in seconds
final_time = 10.  # final time in seconds
# The dominant frequency of the Ricker wavelet in KHz.
frequency_peak = 0.42


def compute_lame_parameters(V, v_s, rho):
    return Function(V).interpolate(rho * v_s * v_s)

degree = 4
V = FunctionSpace(mesh, "KMV", degree)
quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree(), "KMV")

print("DOFs", V.dim())
vs_true = Function(V).interpolate(3.2)
rho = Function(V).interpolate(2.6)
vp_true = Function(V).interpolate(5.8)

source_locations = np.array([50.0, 40.0])
receiver_locations = np.array([150.0, 40.0])


def ricker_wavelet(t, fs, amp=1.0):
    ts = 4.0
    t0 = t - ts * np.sqrt(6.0) / (np.pi * fs)
    return (amp * (1.0 - (1.0 / 2.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0)
            * np.exp((-1.0 / 4.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0))

source_mesh = VertexOnlyMesh(mesh, [source_locations])
V_s = FunctionSpace(source_mesh, "DG", 0)

source_cofunction = assemble(TestFunction(V_s) * dx)
q_s = Cofunction(V.dual()).interpolate(source_cofunction)


def sh_wave_equation(vs, rho, dt, V, f, quad_rule):

    u = TrialFunction(V)
    v = TestFunction(V)
    u_np1 = Function(V, name="u_np1") # timestep n+1
    u_n = Function(V, name="u_n") # timestep n
    u_nm1 = Function(V, name="u_nm1") # timestep n-1
    time_term = (u - 2.0 * u_n + u_nm1) / Constant(dt**2) * v * dx(scheme=quad_rule)
    a = vs*vs * dot(grad(u_n), grad(v)) * dx(scheme=quad_rule)
    F = time_term + a
    lin_var = LinearVariationalProblem(lhs(F), rhs(F) + f, u_np1, constant_jacobian=True)
    solver_parameters = {"mat_type": "matfree", "ksp_type": "preonly", "pc_type": "jacobi"}
    solver = LinearVariationalSolver(lin_var, solver_parameters=solver_parameters)
    return solver, u_np1, u_n, u_nm1


receiver_mesh = VertexOnlyMesh(mesh, [receiver_locations])
V_r = FunctionSpace(receiver_mesh, "DG", 0)
print("Running the true data")
true_data_receivers = []
total_steps = int(final_time / dt) + 1
f = Cofunction(V.dual())  # Wave equation forcing term.
mu = compute_lame_parameters(V, vs_true, rho)
solver, u_np1, u_n, u_nm1 = sh_wave_equation(vs_true, rho, dt, V, f, quad_rule)
interpolate_receivers = interpolate(u_np1, V_r)
output_file = VTKFile("outputs/true_data_3.pvd")
start = time.perf_counter()
for step in range(total_steps):
    f.assign(ricker_wavelet(step * dt, frequency_peak)*q_s)
    solver.solve()
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    true_data_receivers.append(assemble(interpolate_receivers))
    if step % 100 == 0:
        print("Writing step %d" % step, flush=True)
        output_file.write(u_np1)
    # # if norm(u_np1) > 1e10:
    #     raise ValueError("The simulation has diverged.")
    #     break

end = time.perf_counter()
execution_time = end - start
print("The forward simulation took", execution_time, "seconds.")
quit()
from firedrake.adjoint import *
continue_annotation()
tape = get_working_tape()
print("Running the optimisation")
f = Cofunction(V.dual())  # Wave equation forcing term.
solver, u_np1, u_n, u_nm1 = sh_wave_equation(vs_true, rho, dt, V, f, quad_rule)
interpolate_receivers = interpolate(u_np1, V_r)
# J_val = 0.0
# misfit = Function(V_r, name="misfit")
output_file = VTKFile("outputs/guess_data.pvd")
start = time.perf_counter()
for step in tape.timestepper(iter(range(total_steps))):
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    f.assign(ricker_wavelet(step * dt, frequency_peak) * q_s)
    solver.solve()
    assemble(interpolate_receivers)
    if step % 500 == 0:
        print("Writing step %d" % step, flush=True)
        output_file.write(u_np1)
    # misfit = misfit.assign(assemble(interpolate_receivers) - true_data_receivers[step])
    # J_val += 0.5 * assemble(inner(misfit, misfit) * dx)

end = time.perf_counter()
print("The annotation plus forward simulation took", end - start, "seconds.")