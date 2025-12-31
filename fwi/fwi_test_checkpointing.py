from tools import model_elastic as model
from tools import model_interpolate
from firedrake import *
import finat
from firedrake.__future__ import interpolate
import numpy as np
import time

M = 1
my_ensemble = Ensemble(COMM_WORLD, M)
num_sources = my_ensemble.ensemble_comm.size
source_number = my_ensemble.ensemble_comm.rank

mesh = Mesh(
    model["mesh"]["meshfile"], comm=my_ensemble.comm,
    distribution_parameters={
                "overlap_type": (DistributedMeshOverlapType.NONE, 0)
            }, name="mesh"
)
mesh.coordinates.checkpoint_time_dependent = False
element = FiniteElement("KMV", mesh.ufl_cell(), degree=4)

V = VectorFunctionSpace(mesh, element)
V0 = FunctionSpace(mesh, element)
print("DOFs", V.dim())
vp_true = model_interpolate(model, mesh, V0, name="vp_true")
vp_guess = model_interpolate(
    model, mesh, V0, name="vp_true", smoth=True, sigma=100,
    checkpoint_time_dependent=False)
vs_true = model_interpolate(model, mesh, V0, name="vs_true", velocity="vs_truemodel")
vs_guess = model_interpolate(
    model, mesh, V0, name="vs_true", smoth=True, sigma=100, velocity="vs_truemodel",
    checkpoint_time_dependent=False)


rho = model_interpolate(model, mesh, V0, name="rho_true", velocity="density_initmodel")
water = np.where(vp_true.dat.data_ro < 1.51)
vp_guess.dat.data[water] = 1.51
vs_guess.dat.data[water] = 0.0


def compute_lame_parameters(V0, v_p, v_s, rho):
    mu = Function(V0).interpolate(rho * v_s * v_s)
    lamb = Function(V0).interpolate(rho * (v_p * v_p - 2 * v_s * v_s))
    return mu, lamb


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


print("source_number", source_number)
source_mesh = VertexOnlyMesh(mesh, [source_locations[source_number]])
source_mesh.coordinates.checkpoint_time_dependent = False
V_s = VectorFunctionSpace(source_mesh, "DG", 0)

d_s = Function(V_s)
d_s.assign(1.0)

source_cofunction = assemble(inner(d_s, TestFunction(V_s)) * dx)
q_s = Cofunction(V.dual()).interpolate(source_cofunction)

quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree(), "KMV")


def wave_equation_solver(rho, v_s, v_p, source_function, dt, V):
    u = TrialFunction(V)
    v = TestFunction(V)
    u_np1 = Function(V, name="u_np1") # timestep n+1
    u_n = Function(V, name="u_n") # timestep n
    u_nm1 = Function(V, name="u_nm1") # timestep n-1
    mu, lamb = compute_lame_parameters(V0, v_p, v_s, rho)
    def D(w):   
        # strain tensor
        return 0.5 * (grad(w) + grad(w).T)

    # mass matrix 
    m = (rho * inner((u - 2.0 * u_n + u_nm1), v) / Constant(dt ** 2)) * dx(scheme=quad_rule)  # explicit
    # stiffness matrix
    a = lamb * tr(D(u_n)) * tr(D(v)) * dx(scheme=quad_rule) + 2.0 * mu * inner(D(u_n), D(v)) * dx(scheme=quad_rule)
    # get normal and tangent vectors
    n = FacetNormal(mesh)
    t = perp(n)
    C = v_p * outer(n, n) + v_s * outer(t, t)
    nf = inner(C * ((u_n - u_nm1) / dt), v ) * ds # backward-difference scheme
    F = m + a + nf
    lin_var = LinearVariationalProblem(lhs(F), rhs(F) + source_function, u_np1, constant_jacobian=True)
    solver_parameters = {"mat_type": "matfree", "ksp_type": "preonly", "pc_type": "jacobi"}
    solver = LinearVariationalSolver(lin_var,solver_parameters=solver_parameters)
    return solver, u_np1, u_n, u_nm1


print("Running the receivers")
receiver_mesh = VertexOnlyMesh(mesh, receiver_locations)
V_r = VectorFunctionSpace(receiver_mesh, "DG", 0)
print("Running the true data")
true_data_receivers = []
total_steps = int(final_time / dt) + 1
f = Cofunction(V.dual())  # Wave equation forcing term.
solver, u_np1, u_n, u_nm1 = wave_equation_solver(rho, vs_true, vp_true, f, dt, V)
interpolate_receivers = interpolate(u_np1, V_r)
for step in range(total_steps):
    f.assign(ricker_wavelet(step * dt, frequency_peak) * q_s)
    solver.solve()
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    true_data_receivers.append(assemble(interpolate_receivers))
    if step % model["timeaxis"]["fspool"] == 0:
        print("Writing step %d" % step, flush=True)
    #     output_file.write(u_np1)
    # if norm(u_np1) > 1e10:
    #     raise ValueError("The simulation has diverged.")
    #     break


from firedrake.adjoint import *
continue_annotation()
tape = get_working_tape()
from checkpoint_schedules import Revolve, MixedCheckpointSchedule, StorageType, SingleMemoryStorageSchedule
tape.enable_checkpointing(
    # SingleMemoryStorageSchedule(),
    # MixedCheckpointSchedule(total_steps, 40, storage=StorageType.RAM),
    Revolve(total_steps, 40),
    gc_timestep_frequency=50)
print("Running the optimisation")
tape.progress_bar = ProgressBar
f = Cofunction(V.dual())  # Wave equation forcing term.
solver, u_np1, u_n, u_nm1 = wave_equation_solver(rho, vs_guess, vp_guess, f, dt, V)
interpolate_receivers = interpolate(u_np1, V_r)
J_val = 0.0
misfit = Function(V_r, name="misfit")
for step in tape.timestepper(iter(range(total_steps))):
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    f.assign(ricker_wavelet(step * dt, frequency_peak) * q_s)
    solver.solve()
    misfit = misfit.assign(
        assemble(interpolate_receivers) - true_data_receivers[step])
    J_val += 0.5 * assemble(inner(misfit, misfit) * dx)
    if step % model["timeaxis"]["fspool"] == 0:
        print("Writing step %d" % step, flush=True)
    #     output_file.write(u_np1)
    # if J_val > 1e10:
    #     raise ValueError("The simulation has diverged.")
    #     break

start = time.time()
J_hat = EnsembleReducedFunctional(
    J_val, [Control(vp_guess), Control(vs_guess)], my_ensemble)
J_hat.derivative()
end = time.time()
print("Time to run the optimisation", end - start)