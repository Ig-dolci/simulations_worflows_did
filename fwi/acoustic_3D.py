from tools import model_interpolate
from firedrake import *
import finat
import FIAT
from firedrake.__future__ import interpolate
import numpy as np
import time
from mpi4py import MPI
# read a hdf5 file
M = 4
my_ensemble = Ensemble(MPI.COMM_WORLD, M)
num_sources = my_ensemble.ensemble_comm.size
source_number = my_ensemble.ensemble_comm.rank
mesh = ExtrudedMesh(
    RectangleMesh(32, 32, 2, 2, comm=my_ensemble.comm, quadrilateral=True),
    16, layer_height=2/16
)
# mesh = CubeMesh(32, 32, 16, 2., comm=my_ensemble.comm, hexahedral=True)
dt = 0.001  # time step in seconds
final_time = 1.0  # final time in seconds
# The dominant frequency of the Ricker wavelet in KHz.
frequency_peak = 0.1


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


degree = 4
element = FiniteElement('CG', mesh.ufl_cell(), degree=degree, variant='spectral')
V = FunctionSpace(mesh, element)
quad_rule = gauss_lobatto_legendre_cube_rule(dimension=3, degree=degree)
# quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree(), "KMV")



print("DOFs", V.dim())
vs_true = Function(V).interpolate(3.2)
rho = Function(V).interpolate(1.8)
vp_true = Function(V).interpolate(2.)
# VTKFile("outputs/true_model.pvd").write(vp_true)

print("Creating the source and receiver locations")
source_locations = np.array([1.32, 0.82, 0.72])
receiver_locations = np.array([1.32, 1.82, 0.72])


def ricker_wavelet(t, fs, amp=1.0):
    ts = 4.0
    t0 = t - ts * np.sqrt(6.0) / (np.pi * fs)
    return (amp * (1.0 - (1.0 / 2.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0)
            * np.exp((-1.0 / 4.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0))


print("Creating the source wavelet")
source_mesh = VertexOnlyMesh(mesh, [source_locations])
V_s = FunctionSpace(source_mesh, "DG", 0)

source_cofunction = assemble(TestFunction(V_s) * dx)
q_s = Cofunction(V.dual()).interpolate(source_cofunction)
# q_s.riesz_representation()


def sh_wave_equation(vp, dt, V, f, quad_rule, rho=1.8):

    u = TrialFunction(V)
    v = TestFunction(V)
    u_np1 = Function(V, name="u_np1")  # timestep n+1
    u_n = Function(V, name="u_n")  # timestep n
    u_nm1 = Function(V, name="u_nm1")  # timestep n-1
    time_term = rho * (u - 2.0 * u_n + u_nm1) / Constant(dt**2) * v * dx(scheme=quad_rule)
    a = vp * vp * dot(grad(u_n), grad(v)) * dx(scheme=quad_rule)
    F = time_term + a
    lin_var = LinearVariationalProblem(lhs(F), rhs(F) + f, u_np1, constant_jacobian=True)
    solver_parameters = {
        "mat_type": "matfree", "ksp_type": "preonly", "pc_type": "jacobi",
        "pc_factor_mat_solver_type" : "mumps"
    }
    solver = LinearVariationalSolver(lin_var, solver_parameters=solver_parameters)
    return solver, u_np1, u_n, u_nm1


print("Creating the receiver locations")
receiver_mesh = VertexOnlyMesh(mesh, [receiver_locations])
V_r = FunctionSpace(receiver_mesh, "DG", 0)
print("Running the true data")
true_data_receivers = []
total_steps = int(final_time / dt) + 1
f = Cofunction(V.dual())  # Wave equation forcing term.
solver, u_np1, u_n, u_nm1 = sh_wave_equation(vp_true, dt, V, f, quad_rule)
interpolate_receivers = interpolate(u_np1, V_r)
output_file = VTKFile("outputs/true_data_3D.pvd")
solver.solve()
start = time.perf_counter()
for step in range(total_steps):
    f.assign(ricker_wavelet(step * dt, frequency_peak)*q_s)
    solver.solve()
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    assemble(interpolate_receivers)
    if step % 100 == 0:
        print("Writing step %d" % step, flush=True)
        output_file.write(u_np1)
    # if norm(u_np1) > 1e10:
    #     raise ValueError("The simulation has diverged.")
    #     break

end = time.perf_counter()
execution_time = end - start
print("The forward simulation took", execution_time, "seconds.")
