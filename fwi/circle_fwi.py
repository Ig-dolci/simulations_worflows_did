from firedrake import *
from firedrake.adjoint import *
from pyadjoint import TAOSolver, MinimizationProblem
import numpy as np
import warnings
import finat
# use parser to get the number of sources
from argparse import ArgumentParser
from firedrake.__future__ import interpolate
M = 1
my_ensemble = Ensemble(COMM_WORLD, M)
num_sources = my_ensemble.ensemble_comm.size
source_number = my_ensemble.ensemble_comm.rank
mesh = UnitSquareMesh(20, 20, comm=my_ensemble.comm)

source_locations = np.linspace((0.3, 0.1), (0.7, 0.1), num_sources)
receiver_locations = np.linspace((0.2, 0.9), (0.8, 0.9), 20)
dt = 0.001  # time step in seconds
final_time = 1.0  # final time in seconds
frequency_peak = 7.0  # The dominant frequency of the Ricker wavelet in Hz.

V = FunctionSpace(mesh, "KMV", 3)
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


def wave_equation_solver(c, source_function, dt, V):
    u = TrialFunction(V)
    v = TestFunction(V)
    u_np1 = Function(V) # timestep n+1
    u_n = Function(V) # timestep n
    u_nm1 = Function(V) # timestep n-1
    # Quadrature rule for lumped mass matrix.
    quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree(), "KMV")
    m = (1 / (c * c))
    time_term =  m * (u - 2.0 * u_n + u_nm1) / Constant(dt**2) * v * dx(scheme=quad_rule)
    nf = (1 / c) * ((u_n - u_nm1) / dt) * v * ds
    a = dot(grad(u_n), grad(v)) * dx(scheme=quad_rule)
    F = time_term + a + nf
    lin_var = LinearVariationalProblem(lhs(F), rhs(F) + source_function, u_np1)
    # Since the linear system matrix is diagonal, the solver parameters are set to construct a solver,
    # which applies a single step of Jacobi preconditioning.
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

for step in range(total_steps):
    f.assign(ricker_wavelet(step * dt, frequency_peak) * q_s)
    solver.solve()
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    true_data_receivers.append(assemble(interpolate_receivers))


c_guess = Function(V).interpolate(1.5)

continue_annotation()
tape = get_working_tape()
# from checkpoint_schedules import Revolve
# tape.enable_checkpointing(Revolve(total_steps, 10))
f = Cofunction(V.dual())  # Wave equation forcing term.
solver, u_np1, u_n, u_nm1 = wave_equation_solver(c_guess, f, dt, V)
interpolate_receivers = interpolate(u_np1, V_r)
J_val = 0.0
for step in tape.timestepper(iter(range(total_steps))):
    f.assign(ricker_wavelet(step * dt, frequency_peak) * q_s)
    solver.solve()
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    guess_receiver = assemble(interpolate_receivers)
    misfit = guess_receiver - true_data_receivers[step]
    J_val += 0.5 * assemble(inner(misfit, misfit) * dx)

# We now instantiate :class:`~.EnsembleReducedFunctional`::
# get_working_tape().progress_bar = ProgressBar
J_hat = EnsembleReducedFunctional(J_val, Control(c_guess), my_ensemble)


lb = 1.5
up = 2.0

problem = MinimizationProblem(J_hat, bounds=(lb, up))
solver = TAOSolver(problem, {"tao_type": "blmvm", "tao_max_it": 20}, comm=my_ensemble.comm)
outfile = VTKFile("c_optimised_circle.pvd")


def convergence_tracker(tao, *, gatol=1.0e-7, max_its=20):
    its, _, res, _, _, _ = tao.getSolutionStatus()
    outfile.write(J_hat.controls[0].control)
    if res < gatol or its >= max_its:
        tao.setConvergedReason(PETSc.TAO.ConvergedReason.CONVERGED_USER)
    else:
        tao.setConvergedReason(PETSc.TAO.ConvergedReason.CONTINUE_ITERATING)


solver.tao.setConvergenceTest(convergence_tracker)
c_optimised = solver.solve()
