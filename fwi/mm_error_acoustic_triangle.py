# /from tools import model_interpolate
from tools import model_elastic as model
from tools import read_segy, parameter_interpolate
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
file_name = "comp_performance/rec_data_adapt7_acoustic.npy"
my_ensemble = Ensemble(MPI.COMM_WORLD, M)
num_sources = my_ensemble.ensemble_comm.size
source_number = my_ensemble.ensemble_comm.rank
mesh = Mesh(
    model["mesh"]["meshfile"], comm=my_ensemble.comm,
    distribution_parameters={
                "overlap_type": (DistributedMeshOverlapType.NONE, 0)
            }, name="mesh"
)


dt = model["timeaxis"]["dt"]  # time step in seconds
final_time = model["timeaxis"]["tf"]  # final time in seconds
# The dominant frequency of the Ricker wavelet in KHz.
frequency_peak = model["acquisition"]["frequency"]


degree = 4
element = FiniteElement('KMV', mesh.ufl_cell(), degree=4)
V = FunctionSpace(mesh, element)
print("DOFs", V.dim())

vp = read_segy(model["mesh"]["vp"])
vp_true = parameter_interpolate(model, V, vp.T, l_grid=1.25/1000, name="vp_true")
VTKFile("outputs/vp_true.pvd").write(vp_true)

source_locations = [9, -0.1]
receiver_locations = [10, -0.1]


def ricker_wavelet(t, fs, amp=1.0):
    ts = 0.0
    t0 = t- ts * np.sqrt(6.0) / (np.pi * fs)
    return (amp * (1.0 - (1.0 / 2.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0)
            * np.exp((-1.0 / 4.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0))

source_mesh = VertexOnlyMesh(mesh, [source_locations])
V_s = FunctionSpace(source_mesh, "DG", 0)

source_cofunction = assemble(TestFunction(V_s) * dx)
q_s = Cofunction(V.dual()).interpolate(source_cofunction)


def sh_wave_equation(vp, dt, V, f):

    u = TrialFunction(V)
    v = TestFunction(V)
    u_np1 = Function(V, name="u_np1") # timestep n+1
    u_n = Function(V, name="u_n") # timestep n
    u_nm1 = Function(V, name="u_nm1") # timestep n-1
    time_term = (u - 2.0 * u_n + u_nm1) / Constant(dt**2) * v * dx(scheme="KMV", degree=degree)
    a = vp*vp * dot(grad(u_n), grad(v)) * dx(scheme="KMV", degree=degree)
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
solver, u_np1, u_n, u_nm1 = sh_wave_equation(vp_true, dt, V, f)
interpolate_receivers = interpolate(u_np1, V_r)
output_file = VTKFile("outputs/true_data_2D.pvd")
start = time.perf_counter()
for step in range(total_steps):
    f.assign(ricker_wavelet(step * dt, frequency_peak)*q_s)
    solver.solve()
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    true_data_receivers.append(assemble(interpolate_receivers).dat.data)
    if step % 100 == 0:
        print("Writing step %d" % step, flush=True)
        output_file.write(u_np1)
    if norm(u_np1) > 1e10:
        raise ValueError("The simulation has diverged.")
        break

end = time.perf_counter()
print("Time taken", end - start)

# np.save(
#     file_name,
#     true_data_receivers)