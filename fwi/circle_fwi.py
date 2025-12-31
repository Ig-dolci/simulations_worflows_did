from firedrake import *
from firedrake.adjoint import *
from pyadjoint import TAOSolver, MinimizationProblem
from ROL.firedrake_vector import FiredrakeVector as FeVector
import ROL
import numpy as np
import warnings
# Hide warnings
warnings.filterwarnings("ignore")
import finat
import time
from firedrake.__future__ import interpolate
M = 1
my_ensemble = Ensemble(COMM_WORLD, M)
num_sources = my_ensemble.ensemble_comm.size
source_number = my_ensemble.ensemble_comm.rank
mesh = UnitSquareMesh(100, 100, comm=my_ensemble.comm)
mesh.coordinates.checkpoint_time_dependent = False
source_locations = np.linspace((0.3, 0.1), (0.7, 0.1), num_sources)
receiver_locations = np.linspace((0.2, 0.8), (0.8, 0.8), 20)
# mesh = Mesh(
#     '/Users/ddolci/work/firedrake_new/spyro_meshes/square.msh',
#     comm=my_ensemble.comm,
#     distribution_parameters={
#                 "overlap_type": (DistributedMeshOverlapType.NONE, 0)
#             }, name="mesh"
# )
# source_locations = np.linspace((-0.1, 0.2), (-0.1, 0.8), num_sources)
# receiver_locations = np.linspace((-0.8, 0.2), (-0.8, 0.8), 20)
dt = 0.001  # time step in seconds
final_time = 1.0  # final time in seconds
frequency_peak = 0.42  # The dominant frequency of the Ricker wavelet in Hz.

V = FunctionSpace(mesh, "KMV", 1)
x, z = SpatialCoordinate(mesh)
c_true = Function(V).interpolate(1.75 + 0.25 * tanh(200 * (0.125 - sqrt((x - 0.5) ** 2 + (z - 0.5) ** 2))))
# Transform the true model to the dual space.
measure_options = {"scheme": finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree(), "KMV")}

# VTKFile("outputs/true_model.pvd").write(c_true)

def ricker_wavelet(t, fs, amp=1.0):
    ts = 1.5
    t0 = t - ts * np.sqrt(6.0) / (np.pi * fs)
    return (amp * (1.0 - (1.0 / 2.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0)
            * np.exp((-1.0 / 4.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0))


source_mesh = VertexOnlyMesh(mesh, [source_locations[source_number]])
source_mesh.coordinates.checkpoint_time_dependent = False
V_s = FunctionSpace(source_mesh, "DG", 0)
d_s = Function(V_s)
d_s.interpolate(1.0)
source_cofunction = assemble(d_s * TestFunction(V_s) * dx)
q_s = Cofunction(V.dual()).interpolate(source_cofunction)
quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree(), "KMV")

def wave_equation_solver(c, source_function, dt, V):
    u = TrialFunction(V)
    v = TestFunction(V)
    u_np1 = Function(V, name="u_np1") # timestep n+1
    u_n = Function(V, name="u_n") # timestep n
    u_nm1 = Function(V, name="u_nm1") # timestep n-1
    time_term = (u - 2.0 * u_n + u_nm1) / Constant(dt**2) * v * dx(scheme=quad_rule)
    nf = ((u_n - u_nm1) / dt) * v * ds
    a = c * c * dot(grad(u_n), grad(v)) * dx(scheme=quad_rule)
    F = time_term + a + nf
    lin_var = LinearVariationalProblem(lhs(F), rhs(F) + source_function, u_np1, constant_jacobian=True)
    # Since the linear system matrix is diagonal, the solver parameters are set to construct a solver,
    # which applies a single step of Jacobi preconditioning.
    solver_parameters = {"mat_type": "matfree", "ksp_type": "preonly", "pc_type": "jacobi"}
    solver = LinearVariationalSolver(lin_var, solver_parameters=solver_parameters)
    return solver, u_np1, u_n, u_nm1


receiver_mesh = VertexOnlyMesh(mesh, receiver_locations)
receiver_mesh.coordinates.checkpoint_time_dependent = False
V_r = FunctionSpace(receiver_mesh, "DG", 0)
true_data_receivers = []
total_steps = int(final_time / dt) + 1
f = Cofunction(V.dual())  # Wave equation forcing term.
solver, u_np1, u_n, u_nm1 = wave_equation_solver(c_true, f, dt, V)
interpolate_receivers = interpolate(u_np1, V_r)
start = time.time()
solution = []
for step in range(total_steps):
    f.assign(ricker_wavelet(step * dt, frequency_peak) * q_s)
    solver.solve()
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    # solution.append(u_n.copy(deepcopy=True))
    true_data_receivers.append(assemble(interpolate_receivers))

end = time.time()
print("Time taken for the forward simulation: ", end - start)

VTKFile("outputs/true_data.pvd").write(u_np1)

c_guess = Function(V, name="c_guess").interpolate(1.5)
continue_annotation()
tape = get_working_tape()
from checkpoint_schedules import (
    Revolve, MixedCheckpointSchedule, StorageType, SingleMemoryStorageSchedule,
    NoneCheckpointSchedule)
tape.enable_checkpointing(
    MixedCheckpointSchedule(total_steps, 100, storage=StorageType.RAM),
    gc_timestep_frequency=100)
# tape.enable_checkpointing(
#     Revolve(total_steps, 20), gc_timestep_frequency=100)
# tape.enable_checkpointing(SingleMemoryStorageSchedule(), gc_timestep_frequency=100)
mfn_parameters = {"mfn_type": "krylov", "mfn_tol": 1.0e-12}
measure_options = {"scheme": quad_rule}
f = Cofunction(V.dual(), name="f")
solver, u_np1, u_n, u_nm1 = wave_equation_solver(c_guess, f, dt, V)
interpolate_receivers = interpolate(u_np1, V_r)
J_val = 0.0
for step in tape.timestepper(iter(range(total_steps))):
    f.assign(ricker_wavelet(step * dt, frequency_peak) * q_s)
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    solver.solve()

    guess_receiver = assemble(interpolate_receivers)
    misfit = guess_receiver - true_data_receivers[step]
    J_val += 0.5 * assemble(inner(misfit, misfit) * dx)
    if J_val > 1e10:
        raise ValueError("The simulation has diverged.")
        break

control = Control(c_guess)
J_hat = EnsembleReducedFunctional(J_val, control, my_ensemble)
J_hat(c_guess)
J_hat.derivative()
print("The value of the functional is: ", J_hat.functional)
quit()
with stop_annotating():
    # Water closer to the source is less important for the inversion.
    water = np.where(c_true.dat.data_ro < 1.51)
    c_computed = Function(V, name="c_optimised")
    outfile = VTKFile("c_optimised_circle.pvd")

    def regularise_gradient(functional, dJ, controls, gamma=1.0e-6):
        c_computed.assign(controls[0])
        # controls[0] = transform(controls[0], TransformType.DUAL, mfn_parameters=mfn_parameters, measure_options=measure_options)
        # Check if dJ is a NaN
        if np.isnan(dJ[0].dat.data_ro).any():
            raise ValueError("The gradient is NaN")
        print("The gradient is fine.", flush=True)
        outfile.write(c_computed)
        return dJ

    # def eval_cb_pre(controls):
    #     controls.assign(
    #         transform(
    #             controls, TransformType.PRIMAL, mfn_parameters=mfn_parameters,
    #             measure_options=measure_options))

    # J_hat = EnsembleReducedFunctional(
    #     J_val, control, my_ensemble, derivative_cb_post=regularise_gradient)
    J_hat = EnsembleReducedFunctional(J_val, control, my_ensemble)
    J_hat.derivative()
    # taylor_test(J_hat, c_guess, Function(V).interpolate(1.0))
    # lb = 1.5
    # up = 2.0

    # problem = MinimizationProblem(J_hat, bounds=(lb, up))
    # params_dict = {
    #     'General': {
    #         'Secant': {'Type': 'Limited-Memory BFGS', 'Maximum Storage': 10}},
    #     'Step': {
    #         'Type': 'Augmented Lagrangian',
    #         'Line Search': {
    #             'Descent Method': {
    #                 'Type': 'Quasi-Newton Step'
    #             }
    #         },
    #         'Augmented Lagrangian': {
    #             'Subproblem Step Type': 'Line Search',
    #             'Subproblem Iteration Limit': 10
    #         }
    #     },
    #     'Status Test': {
    #         'Gradient Tolerance': 1e-7,
    #         'Iteration Limit': 10
    #     }
    # }

    # solver_parameters = {"mat_type": "matfree", "ksp_type": "preonly", "pc_type": "jacobi"}
    # solver = ROLSolver(problem, params_dict, inner_product="L2",
    #                 inner_product_solver_opts={
    #                     "solver_options": {"solver_parameters" : solver_parameters},
    #                     "measure_options": {"scheme": quad_rule}})
    # rho_opt = solver.solve()
