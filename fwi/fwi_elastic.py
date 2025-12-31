from tools import model_elastic as model
from tools import model_interpolate
from tools import damp_functions
from firedrake import *
import finat
import FIAT
from firedrake.__future__ import interpolate
import numpy as np
import scipy.ndimage
from pyadjoint import TAOSolver, MinimizationProblem

M = 2
my_ensemble = Ensemble(COMM_WORLD, M)
num_sources = my_ensemble.ensemble_comm.size
source_number = my_ensemble.ensemble_comm.rank
# mesh = RectangleMesh(324, 95, 12, 3.5, quadrilateral=True)

# mesh.coordinates.dat.data[:, 0] *= -1
# element = FiniteElement('CG', mesh.ufl_cell(), degree=3, variant='spectral')
# mesh = Mesh(
#     model["mesh"]["meshfile"], comm=my_ensemble.comm,
#     distribution_parameters={
#                 "overlap_type": (DistributedMeshOverlapType.NONE, 0)
#             }, name="mesh"
# )
# mesh = checkpointable_mesh(mesh)
with CheckpointFile("outputs/elastic_optimised.h5", 'r', comm=my_ensemble.comm) as afile:
    mesh = afile.load_mesh("mesh")
# mesh.coordinates.dat.data[:] *= 1000
# mesh.coordinates.dat.data[:, 1] = - mesh.coordinates.dat.data[:, 1] * 1000 - model["mesh"]["zmin"]
# for i in range(len(mesh.coordinates.dat.data)):
#     mesh.coordinates.dat.data[i, 0] = mesh.coordinates.dat.data[i, 0] * 1000 + model["mesh"]["xmin"]

element = FiniteElement("KMV", mesh.ufl_cell(), degree=1)

# element = FiniteElement('CG', mesh.ufl_cell(), degree=4, variant='spectral')


def gauss_lobatto_legendre_line_rule(degree):
    fiat_make_rule = FIAT.quadrature.GaussLobattoLegendreQuadratureLineRule
    fiat_rule = fiat_make_rule(FIAT.ufc_simplex(1), degree + 1)
    finat_ps = finat.point_set.GaussLobattoLegendrePointSet
    points = finat_ps(fiat_rule.get_points())
    weights = fiat_rule.get_weights()
    return finat.quadrature.QuadratureRule(points, weights)


# Quadrature rule for lumped mass matrix.
def gauss_lobatto_legendre_cube_rule(dimension, degree):
    make_tensor_rule = finat.quadrature.TensorProductQuadratureRule
    result = gauss_lobatto_legendre_line_rule(degree)
    for _ in range(1, dimension):
        line_rule = gauss_lobatto_legendre_line_rule(degree)
        result = make_tensor_rule([result, line_rule])
    return result

# element = FiniteElement('CG', mesh.ufl_cell(), degree=3, variant='spectral')

V = VectorFunctionSpace(mesh, element)
V0 = FunctionSpace(mesh, element)
print("DOFs", V.dim())
# quit()
vp_true = model_interpolate(model, mesh, V0, name="vp_true")
VTKFile("outputs/vp_true.pvd").write(vp_true)
# vp_guess = model_interpolate(
#     model, mesh, V0, name="vp_true", smoth=True, sigma=150,
#     checkpoint_time_dependent=False)
vs_true = model_interpolate(model, mesh, V0, name="vs_true", velocity="vs_truemodel")
vs_guess = model_interpolate(
    model, mesh, V0, name="vs_true", smoth=True, sigma=150, velocity="vs_truemodel",
    checkpoint_time_dependent=False)

with CheckpointFile("outputs/elastic_optimised.h5", 'r', comm=my_ensemble.comm) as afile:
    # mesh = afile.load_mesh("mesh")
    vp_guess = afile.load_function(mesh, "vp_optimised")

VTKFile("outputs/vp_true.pvd").write(vp_true)
VTKFile("outputs/vs_true.pvd").write(vs_true)
VTKFile("outputs/vp_guess.pvd").write(vp_guess)
VTKFile("outputs/vs_guess.pvd").write(vs_guess)
rho = model_interpolate(model, mesh, V0, name="rho_true", velocity="density_initmodel")
rho_guess = model_interpolate(
    model, mesh, V0, name="rho_true", smoth=True, sigma=100, velocity="density_initmodel",
    checkpoint_time_dependent=False)
VTKFile("outputs/rho_true.pvd").write(rho)
VTKFile("outputs/rho_guess.pvd").write(rho_guess)

water = np.where(vp_true.dat.data_ro < 1.51)
vp_guess.dat.data[water] = 1.51
vs_guess.dat.data[water] = 0.0
rho_guess.dat.data[water] = 1.0

sigmax, sigmaz = damp_functions(model, V0, mesh)

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
d_s.assign(0.05291 * 0.05291 * 1000)

source_cofunction = assemble(inner(d_s, TestFunction(V_s)) * dx)
q_s = Cofunction(V.dual()).interpolate(source_cofunction)
q_s = q_s.riesz_representation(riesz_map="L2")
quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree(), "KMV")
# quad_rule = gauss_lobatto_legendre_cube_rule(dimension=2, degree=4)


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
    damp = (sigmax + sigmaz) * inner(((u - u_nm1) / Constant(2.0 * dt)) , v) * dx(scheme=quad_rule)
    F = m + a + damp - inner(source_function , v) * dx(scheme=quad_rule)
    lin_var = LinearVariationalProblem(lhs(F), rhs(F), u_np1, constant_jacobian=True)
    solver_parameters = {"mat_type": "matfree", "ksp_type": "preonly", "pc_type": "jacobi"}
    solver = LinearVariationalSolver(lin_var,solver_parameters=solver_parameters)
    return solver, u_np1, u_n, u_nm1

print("Running the receivers")
receiver_mesh = VertexOnlyMesh(mesh, receiver_locations)
V_r = VectorFunctionSpace(receiver_mesh, "DG", 0)
print("Running the true data")
true_data_receivers = []
total_steps = int(final_time / dt) + 1
f = Function(V)  # Wave equation forcing term.
solver, u_np1, u_n, u_nm1 = wave_equation_solver(rho, vs_true, vp_true, f, dt, V)
interpolate_receivers = interpolate(u_np1, V_r)
output_file = VTKFile("outputs/true_data_3.pvd")
for step in range(total_steps):
    f.assign(Constant(ricker_wavelet(step * dt, frequency_peak)) * q_s)
    solver.solve()
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    true_data_receivers.append(assemble(interpolate_receivers))
    if step % model["timeaxis"]["fspool"] == 0:
        print("Writing step %d" % step, flush=True)
        output_file.write(u_np1)
    if norm(u_np1) > 1e10:
        raise ValueError("The simulation has diverged.")
        break

from firedrake.adjoint import *
continue_annotation()
tape = get_working_tape()
from checkpoint_schedules import Revolve, MixedCheckpointSchedule, StorageType, SingleDiskStorageSchedule
enable_disk_checkpointing()
tape.enable_checkpointing(
    # SingleMemoryStorageSchedule(),
    MixedCheckpointSchedule(total_steps, 50, storage=StorageType.RAM),
    # Revolve(total_steps, 40),
    gc_timestep_frequency=50)

print("Running the optimisation")
# tape.progress_bar = ProgressBar
f = Function(V)  # Wave equation forcing term.
solver, u_np1, u_n, u_nm1 = wave_equation_solver(rho_guess, vs_guess, vp_guess, f, dt, V)
interpolate_receivers = interpolate(u_np1, V_r)

J_val = 0.0
misfit = Function(V_r, name="misfit")
output_file = VTKFile("outputs/guess_data.pvd")

for step in tape.timestepper(iter(range(total_steps))):
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    f.assign(Constant(ricker_wavelet(step * dt, frequency_peak)) * q_s)
    solver.solve()
    misfit = misfit.assign(
        assemble(interpolate_receivers) - true_data_receivers[step])
    J_val += 0.5 * assemble(inner(misfit, misfit) * dx)
    if step % model["timeaxis"]["fspool"] == 0:
        print("Writing step %d" % step, flush=True)
        output_file.write(u_np1)
    if J_val > 1e10:
        raise ValueError("The simulation has diverged.")
        break


outfile_vp = VTKFile("outputs/c_optimised_vp.pvd")
outfile_vs = VTKFile("outputs/c_optimised_vs.pvd")
outfile_rho = VTKFile("outputs/c_optimised_rho.pvd")

vp_optimised = Function(V0, name="vp_optimised")
vs_optimised = Function(V0, name="vs_optimised")
rho_optimised = Function(V0, name="rho_optimised")


def regularise_gradient(functional, dJ, controls, regularise=True, gamma=1.0e-2):
    """Tikhonov regularization"""
    print("Regularising gradient", flush=True)
    vp_optimised.assign(controls[0])
    vs_optimised.assign(controls[1])
    rho_optimised.assign(controls[2])
    # outfile_vp.write(vp_optimised)
    # outfile_vs.write(vs_optimised)
    # outfile_rho.write(rho_optimised)
    with CheckpointFile("outputs/elastic_optimised.h5", 'w', comm=my_ensemble.comm) as afile:
        afile.save_mesh(mesh) 
        afile.save_function(vp_optimised)
        afile.save_function(vs_optimised)
        afile.save_function(rho_optimised)
    # dJ_backup = dJ.copy()
    # breakpoint()
    if not regularise:
        return dJ
    # for i0, g in enumerate(dJ):
    #     g.dat.data_wo_with_halos[water] = 0.0
        # g_cof = g.riesz_representation(riesz_map="l2")
        # g = g._ad_convert_riesz(g_cof, options=options)

    # VTKFile("outputs/grad_0.pvd").write(dJ[0])
    # VTKFile("outputs/grad_1.pvd").write(dJ[1])
    # VTKFile("outputs/grad_2.pvd").write(dJ[2])
    return dJ


J_hat = EnsembleReducedFunctional(J_val, [Control(vp_guess), Control(vs_guess), Control(rho_guess)],
                                  my_ensemble, derivative_cb_post=regularise_gradient)
# J_hat.derivative()
taylor_test(J_hat, [vp_guess, vs_guess, rho_guess], [Function(V0).assign(0.1), Function(V0).assign(0.1), Function(V0).assign(0.1)])


# solver_parameters = {"mat_type": "matfree", "ksp_type": "preonly", "pc_type": "jacobi"}
# vs_bound = [0.0, 2.7]
# vp_bound = [1.5, 4.5]
# rho_bound = [1.0, 3.0]


# problem = MinimizationProblem(J_hat, bounds=(vp_bound, vs_bound, rho_bound))
# params = {
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
#         'Iteration Limit': 15
#     }
# }

# solver = ROLSolver(problem, params, inner_product="L2",
#                 inner_product_solver_opts={
#                     "solver_options": {"solver_parameters": solver_parameters},
#                     "measure_options": {"scheme": quad_rule}})
# rho_opt = solver.solve()
