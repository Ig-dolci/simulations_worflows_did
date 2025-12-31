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
my_ensemble = Ensemble(MPI.COMM_WORLD, M)
num_sources = my_ensemble.ensemble_comm.size
source_number = my_ensemble.ensemble_comm.rank
mesh = Mesh(
    model["mesh"]["meshfile"], comm=my_ensemble.comm,
    distribution_parameters={
                "overlap_type": (DistributedMeshOverlapType.NONE, 0)
            }, name="mesh"
)
mesh.coordinates.dat.data[:] *= 1000

dt = model["timeaxis"]["dt"]  # time step in seconds
final_time = model["timeaxis"]["tf"]  # final time in seconds
# The dominant frequency of the Ricker wavelet in KHz.
frequency_peak = model["acquisition"]["frequency"]


degree = 4
element = FiniteElement('KMV', mesh.ufl_cell(), degree=degree)
V = VectorFunctionSpace(mesh, element)
quad_rule = finat.quadrature.make_quadrature(
    V.finat_element.cell, V.ufl_element().degree(), "KMV")

V = FunctionSpace(mesh, element)
print("DOFs", V.dim())
vp = read_segy(model["mesh"]["vp"])
vp_true = parameter_interpolate(model, V, vp.T, l_grid=1.25, name="vp_true")
vp_guess = parameter_interpolate(
    model, V, vp.T, l_grid=1.25, name="vp_true", smoth_par=True)
VTKFile("outputs/vp_true.pvd").write(vp_true)
VTKFile("outputs/vp_guess.pvd").write(vp_guess)

source_locations = model["acquisition"]["source_pos"][source_number]
receiver_locations = model["acquisition"]["receiver_locations"]


def ricker_wavelet(t, fs, amp=1.0):
    ts = 0.0
    t0 = t- ts * np.sqrt(6.0) / (np.pi * fs)
    return (amp * (1.0 - (1.0 / 2.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0)
            * np.exp((-1.0 / 4.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0))

source_mesh = VertexOnlyMesh(mesh, [source_locations])
V_s = FunctionSpace(source_mesh, "DG", 0)
d_s = Function(V_s)
d_s.assign(1.0 * 0.0714 * 0.0714 * 1000.)

source_cofunction = assemble(d_s * TestFunction(V_s) * dx)
q_s = Cofunction(V.dual()).interpolate(source_cofunction)

q_s = q_s.riesz_representation("L2")
VTKFile("outputs/source.pvd").write(q_s)
def sh_wave_equation(vp, dt, V, f, quad_rule):

    u = TrialFunction(V)
    v = TestFunction(V)
    u_np1 = Function(V, name="u_np1") # timestep n+1
    u_n = Function(V, name="u_n") # timestep n
    u_nm1 = Function(V, name="u_nm1") # timestep n-1
    time_term = (u - 2.0 * u_n + u_nm1) / Constant(dt**2) * v * dx(scheme=quad_rule)
    a = vp*vp * dot(grad(u_n), grad(v)) * dx(scheme=quad_rule)
    F = time_term + a - f * v * dx(scheme=quad_rule)
    lin_var = LinearVariationalProblem(lhs(F), rhs(F), u_np1, constant_jacobian=True)
    solver_parameters = {"mat_type": "matfree", "ksp_type": "preonly", "pc_type": "jacobi"}
    solver = LinearVariationalSolver(lin_var, solver_parameters=solver_parameters)
    return solver, u_np1, u_n, u_nm1


receiver_mesh = VertexOnlyMesh(mesh, receiver_locations)
V_r = FunctionSpace(receiver_mesh, "DG", 0)
print("Running the true data")
true_data_receivers = []
total_steps = int(final_time / dt) + 1
f = Function(V)  # Wave equation forcing term.
solver, u_np1, u_n, u_nm1 = sh_wave_equation(vp_true, dt, V, f, quad_rule)
interpolate_receivers = interpolate(u_np1, V_r)
output_file = VTKFile("outputs/true_data_2D.pvd")
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
    if norm(u_np1) > 1e10:
        raise ValueError("The simulation has diverged.")
        break


from firedrake.adjoint import *
continue_annotation()
tape = get_working_tape()
from checkpoint_schedules import *

# tape.enable_checkpointing(
#     # SingleMemoryStorageSchedule(),
#     # Revolve(total_steps, 50),
#     MixedCheckpointSchedule(total_steps, 50, storage=StorageType.RAM),
#     gc_timestep_frequency=100)
# store data in a numpy array
f = Function(V)  # Wave equation forcing term.
solver, u_np1, u_n, u_nm1 = sh_wave_equation(vp_guess, dt, V, f, quad_rule)
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
        print("Step %d, J = %e" % (step, J_val), flush=True)
    if J_val > 1e10:
        raise ValueError("The simulation has diverged.")
        break


if COMM_WORLD.rank == 0:
    print("J = %e" % J_val, flush=True)

outfile = VTKFile("outputs/test.pvd")
water = np.where(vp_true.dat.data_ro < 1.51)
c_computed = Function(V, name="test")


def regularise_gradient(functional, dJ, controls, regularise=True, gamma=1*1e-4):
    """Tikhonov regularization"""
    print("Regularising gradient", flush=True)
    c_computed.assign(controls[0])
    np.save("outputs/control.npy", c_computed.dat.data_ro)
    outfile.write(c_computed)
    if not regularise:
        return dJ

    for i0, g in enumerate(dJ):
        g.dat.data_wo_with_halos[water] = 0.0
    return dJ


J_hat = EnsembleReducedFunctional(
    J_val, Control(vp_guess), my_ensemble, derivative_cb_post=regularise_gradient)

lb = 1.
up = 5.0

# taylor_test(J_hat, vp_guess, Function(V).assign(0.1))
# c_optimised = minimize(
#     J_hat, method="L-BFGS-B", options={"disp": True, "maxiter": 10},
#     bounds=(lb, up), derivative_options={"riesz_representation": 'L2'}
# # )
# problem = MinimizationProblem(J_hat, bounds=(lb, up))
# # BQNLS
# # blmvm
# # bnls
# # BNTR
# solver = TAOSolver(problem, {"tao_type": "blmvm", "tao_max_it": 15}, comm=my_ensemble.comm,
#                    convert_options={"riesz_representation": "L2"})


# def convergence_tracker(tao, *, gatol=1.0e-7, max_its=15):
#     its, _, res, _, _, _ = tao.getSolutionStatus()
#     # outfile.write(J_hat.controls[0].control)
#     if res < gatol or its >= max_its:
#         tao.setConvergedReason(PETSc.TAO.ConvergedReason.CONVERGED_USER)
#     else:
#         tao.setConvergedReason(PETSc.TAO.ConvergedReason.CONTINUE_ITERATING)


# solver.tao.setConvergenceTest(convergence_tracker)
# c_optimised = solver.solve()

problem = MinimizationProblem(J_hat, bounds=(lb, up))
params = {
    'General': {
        'Secant': {'Type': 'Limited-Memory BFGS', 'Maximum Storage': 10}},
    'Step': {
        'Type': 'Augmented Lagrangian',
        'Line Search': {
            'Descent Method': {
                'Type': 'Quasi-Newton Step'
            }
        },
        'Augmented Lagrangian': {
            'Subproblem Step Type': 'Line Search',
            'Subproblem Iteration Limit': 10
        }
    },
    'Status Test': {
        'Gradient Tolerance': 1e-7,
        'Iteration Limit': 15
    }
}


class SpyroFunctional(EnsembleReducedFunctional):
    def __init__(self, J, vp, ensemble, source_locations, q_s):
        super().__init__(J, [Control(vp), Control(q_s)], ensemble)
        self.source_locations = source_locations

    def __call__(self, c):
        # Run ensemble reduced functional for a set of source locations
        q_s = self._source()
        super().__call__(c, q_s)

    def derivative(self):
        return compute_gradient(
            self.functional, self.control[0], tape=self.tape)

    def _source(self):
        with stop_annotating():
            source_mesh = VertexOnlyMesh(mesh, [source_locations])
            V_s = FunctionSpace(source_mesh, "DG", 0)
            d_s = Function(V_s)
            d_s.assign(1.0 * 0.0714 * 0.0714 * 1000.)

            source_cofunction = assemble(d_s * TestFunction(V_s) * dx)
            q_s = Cofunction(V.dual()).interpolate(source_cofunction)

            return q_s.riesz_representation("L2")


solver_parameters = {"mat_type": "matfree", "ksp_type": "preonly", "pc_type": "jacobi"}
solver = ROLSolver(problem, params, inner_product="L2")
                #    inner_product_solver_opts={
                #        "solver_options": {"solver_parameters": solver_parameters},
                #        "measure_options": {"scheme": quad_rule}})
rho_opt = solver.solve()
