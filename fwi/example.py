from tools import model_elastic as model
from firedrake import *
import finat
import FIAT
from firedrake.__future__ import interpolate
import numpy as np
import time
from mpi4py import MPI

mesh = CubeMesh(28, 28, 14, 2., hexahedral=False)
dt = 0.001  # time step in seconds
final_time = 1.0  # final time in seconds
# The dominant frequency of the Ricker wavelet in KHz.
frequency_peak = 0.05

degree = 2  # KMV degree
element = FiniteElement('KMV', mesh.ufl_cell(), degree=degree)

V = FunctionSpace(mesh, element)
quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree(), "KMV")

vp_true = Function(V).interpolate(2.)

source_locations = np.array([1.32, 0.82, 0.72])


print("Creating the source wavelet")
source_mesh = VertexOnlyMesh(mesh, [source_locations])
V_s = FunctionSpace(source_mesh, "DG", 0)

source_cofunction = assemble(TestFunction(V_s) * dx)
q_s = Cofunction(V.dual()).interpolate(source_cofunction)


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


total_steps = int(final_time / dt)
f = Cofunction(V.dual())  # Wave equation forcing term.
solver, u_np1, u_n, u_nm1 = sh_wave_equation(vp_true, dt, V, f, quad_rule)

f.assign(q_s)
output_file = VTKFile("outputs/true_data_3D.pvd")
for step in range(total_steps):
    # print time step
    print(f"Time step {step+1}/{total_steps}")
    solver.solve()
    u_nm1.assign(u_n)
    u_n.assign(u_np1)

    if step % 100 == 0:
        print("Writing step %d" % step, flush=True)
        output_file.write(u_np1)

    if norm(u_np1) > 1e10:
        raise ValueError("The simulation has diverged.")
        break