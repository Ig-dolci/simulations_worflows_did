from tools import model, model_interpolate
from firedrake import *
import finat
from firedrake.__future__ import interpolate
import numpy as np
import scipy.ndimage
# read a hdf5 file
M = 1
my_ensemble = Ensemble(COMM_WORLD, M)
num_sources = my_ensemble.ensemble_comm.size
source_number = my_ensemble.ensemble_comm.rank

mesh = Mesh(
    model["path"] + 'inputs/mm1.msh', comm=my_ensemble.comm,
    distribution_parameters={
                "overlap_type": (DistributedMeshOverlapType.NONE, 0)
            }, name="mesh"
)
V = FunctionSpace(mesh, "KMV", 1)
c_true = model_interpolate(model, mesh, V, guess=False, name="c_true")
VTKFile(model["path"] + "outputs/true_model.pvd").write(c_true)
c_guess = Function(V, name="c_guess")
# Apply gaussian smoothing to the true data to create a guess model.

smoothing_true_data = scipy.ndimage.gaussian_filter(c_true.dat.data_ro, sigma=5)
c_guess.dat.data[:] = smoothing_true_data
VTKFile(model["path"] + "outputs/guess_model.pvd").write(c_guess)

with CheckpointFile(model["path"] + "outputs/vel_models.h5", 'w') as afile:
    afile.save_mesh(mesh)  # optional
    afile.save_function(c_true)
    afile.save_function(c_guess)
source_locations = model["acquisition"]["source_pos"]
receiver_locations = model["acquisition"]["receiver_locations"]
frequency_peak = model["acquisition"]["frequency"]
final_time = model["timeaxis"]["tf"]
dt = model["timeaxis"]["dt"]

def ricker_wavelet(t, fs, amp=1.0):
    ts = 1.5
    t0 = t - ts * np.sqrt(6.0) / (np.pi * fs)
    return (amp * (1.0 - (1.0 / 2.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0)
            * np.exp((-1.0 / 4.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0))


source_mesh = VertexOnlyMesh(mesh, [source_locations[source_number]])

V_s = FunctionSpace(source_mesh, "DG", 0)

d_s = Function(V_s)
d_s.assign(1.0)

source_cofunction = assemble(d_s * TestFunction(V_s) * dx)
q_s = Cofunction(V.dual()).interpolate(source_cofunction)



def wave_equation_solver(c, source_function, dt, V):
    u = TrialFunction(V)
    v = TestFunction(V)
    u_np1 = Function(V, name="u_np1") # timestep n+1
    u_n = Function(V, name="u_n") # timestep n
    u_nm1 = Function(V, name="u_nm1") # timestep n-1
    # Quadrature rule for lumped mass matrix.
    quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree(), "KMV")
    time_term = (1 / (c * c)) * (u - 2.0 * u_n + u_nm1) / Constant(dt**2) * v * dx(scheme=quad_rule)
    nf = (1 / c) * ((u_n - u_nm1) / dt) * v * ds
    a = dot(grad(u_n), grad(v)) * dx(scheme=quad_rule)
    F = time_term + a + nf
    lin_var = LinearVariationalProblem(lhs(F), rhs(F) + source_function, u_np1)
    solver_parameters = {"mat_type": "matfree", "ksp_type": "preonly", "pc_type": "jacobi"}
    solver = LinearVariationalSolver(lin_var,solver_parameters=solver_parameters)
    return solver, u_np1, u_n, u_nm1

receiver_mesh = VertexOnlyMesh(mesh, receiver_locations)
V_r = FunctionSpace(receiver_mesh, "DG", 0)

true_data_receivers = []
total_steps = int(final_time / dt) + 1
f = Cofunction(V.dual())  # Wave equation forcing term.
solver, u_np1, u_n, u_nm1 = wave_equation_solver(c_true, f, dt, V)
interpolate_receivers = interpolate(u_np1, V_r)
output_file = VTKFile(model["path"] + "outputs/true_data_3.pvd")
for step in range(total_steps):
    f.assign(ricker_wavelet(step * dt, frequency_peak) * q_s)
    solver.solve()
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    true_data_receivers.append(assemble(interpolate_receivers))
    if step % model["timeaxis"]["fspool"] == 0:
        print("Writing step %d" % step)
        output_file.write(u_np1)
    if norm(u_np1) > 1e10:
        raise ValueError("The simulation has diverged.")
        break


# from firedrake.adjoint import *
# continue_annotation()

f = Cofunction(V.dual())  # Wave equation forcing term.
solver, u_np1, u_n, u_nm1 = wave_equation_solver(c_guess, f, dt, V)
interpolate_receivers = interpolate(u_np1, V_r)
J_val = 0.0
misfit_data = []
output_file1 = VTKFile(model["path"] + "outputs/guess_data_3.pvd")
for step in range(total_steps):
    f.assign(ricker_wavelet(step * dt, frequency_peak) * q_s)
    solver.solve()
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    guess_receiver = assemble(interpolate_receivers)
    misfit = guess_receiver - true_data_receivers[step]
    misfit_data.append(guess_receiver.dat.data_ro - true_data_receivers[step].dat.data_ro)
    J_val += 0.5 * assemble(inner(misfit, misfit) * dx)
    if step % model["timeaxis"]["fspool"] == 0:
        print("Writing step %d" % step)
        output_file1.write(u_np1)
    if J_val > 1e10:
        raise ValueError("The simulation has diverged.")
        break

with CheckpointFile(model["path"] + "outputs/unp1.h5", 'w') as afile:
    afile.save_function(u_np1)
# save the misfit data
np.save(model["path"] + "outputs/misfit_data_3.npy", misfit_data)

# J_hat = EnsembleReducedFunctional(J_val, Control(c_guess), my_ensemble)

# c_optimised = minimize(
#     J_hat, method="L-BFGS-B", options={"disp": True, "maxiter": 1},
#     bounds=(1.5, 3.5), derivative_options={"riesz_representation": 'l2'}
# )