from firedrake import *
import finat
import FIAT
from firedrake.__future__ import interpolate
import numpy as np
import time

M = 1
my_ensemble = Ensemble(COMM_WORLD, M)
num_sources = my_ensemble.ensemble_comm.size
source_number = my_ensemble.ensemble_comm.rank
mesh = RectangleMesh(25, 25, 1., 1.0, comm=my_ensemble.comm, quadrilateral=True)
dt = 0.0001 # time step in seconds
final_time = 0.1  # final time in seconds
# The dominant frequency of the Ricker wavelet in KHz.
frequency_peak = 0.5


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
quad_rule = gauss_lobatto_legendre_cube_rule(dimension=2, degree=degree)
V = VectorFunctionSpace(mesh, element)
V0 = FunctionSpace(mesh, element)
print("DOFs", V.dim())

vs_true = Function(V0).interpolate(1.5)
rho = Function(V0).interpolate(2.5)
vp_true = Function(V0).interpolate(2.7)

source_locations = np.array([0.3, 0.5])
receiver_locations = np.array([0.7, 0.5])


def ricker_wavelet(t, fs, amp=1.0):
    ts = 1.5
    t0 = t - ts * np.sqrt(6.0) / (np.pi * fs)
    return (amp * (1.0 - (1.0 / 2.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0)
            * np.exp((-1.0 / 4.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0))


def compute_lame_parameters(V0, v_p, v_s, rho):
    mu = Function(V0).interpolate(rho * v_s * v_s)
    lamb = Function(V0).interpolate(rho * (v_p * v_p - 2 * v_s * v_s))
    return mu, lamb


source_mesh = VertexOnlyMesh(mesh, [source_locations])
V_s = VectorFunctionSpace(source_mesh, "DG", 0)

d_s = Function(V_s)
d_s.assign(1.0)

source_cofunction = assemble(inner(d_s, TestFunction(V_s)) * dx)
q_s = Cofunction(V.dual()).interpolate(source_cofunction)


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
    F = m + a
    lin_var = LinearVariationalProblem(lhs(F), rhs(F) + source_function, u_np1, constant_jacobian=True)
    solver_parameters = {"mat_type": "matfree", "ksp_type": "preonly", "pc_type": "jacobi"}
    solver = LinearVariationalSolver(lin_var,solver_parameters=solver_parameters)
    return solver, u_np1, u_n, u_nm1



receiver_mesh = VertexOnlyMesh(mesh, [receiver_locations])
V_r = VectorFunctionSpace(receiver_mesh, "DG", 0)
print("Running the true data")
true_data_receivers = []
total_steps = int(final_time / dt) + 1
f = Cofunction(V.dual())  # Wave equation forcing term.
solver, u_np1, u_n, u_nm1 = wave_equation_solver(rho, vs_true, vp_true, f, dt, V)
interpolate_receivers = interpolate(u_np1, V_r)
output_file = VTKFile("outputs/true_data_2D.pvd")
start = time.perf_counter()
for step in range(total_steps):
    f.assign(ricker_wavelet(step * dt, frequency_peak) * q_s)
    solver.solve()
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    true_data_receivers.append(assemble(interpolate_receivers))
    if step % 200 == 0:
        print("Writing step %d" % step, flush=True)
    #     output_file.write(u_np1)
    # if norm(u_np1) > 1e10:
    #     raise ValueError("The simulation has diverged.")
    #     break

end = time.perf_counter()
print("The annotation plus forward simulation took", end - start, "seconds.")