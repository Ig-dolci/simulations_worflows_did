from tools.wave_parameter import parameter_interpolate, read_segy
from tools.model_parameter_elastic import model
from tools import damp_functions
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
file_name = "comp_performance/rec_data_adapt_elastic_water.npy"


mesh = Mesh(
    model["mesh"]["meshfile"], comm=my_ensemble.comm,
    distribution_parameters={
                "overlap_type": (DistributedMeshOverlapType.NONE, 0)
            }, name="mesh"
)

element = FiniteElement('KMV', mesh.ufl_cell(), degree=4)

V = VectorFunctionSpace(mesh, element)
V0 = FunctionSpace(mesh, element)
print("Reading the segys")
vp = read_segy(model["mesh"]["vp"])
vs = read_segy(model["mesh"]["vs"])
rho = read_segy(model["mesh"]["density"])

quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree(), "KMV")

print("DOFs", V.dim())
# quit()
print("Interpolating the parameter vp")
vp_true = parameter_interpolate(model, V0, vp.T, l_grid=1.25/1000, name="vp_true")
print("Interpolating the parameter vs")
vs_true = parameter_interpolate(model, V0, vs.T, l_grid=1.25/1000, name="vs_true")
print("Interpolating the parameter rho")
rho = parameter_interpolate(model, V0, rho.T, l_grid=1.25/1000, name="rho")

VTKFile("outputs/vp_true.pvd").write(vp_true)
VTKFile("outputs/vs_true.pvd").write(vs_true)
VTKFile("outputs/rho_true.pvd").write(rho)

sigmax, sigmaz = damp_functions(model, V0, mesh)


def compute_lame_parameters(V0, v_p, v_s, rho):
    mu = Function(V0).interpolate(rho * v_s * v_s)
    lamb = Function(V0).interpolate(rho * (v_p * v_p - 2 * v_s * v_s))
    return mu, lamb


source_locations = [9, -0.6]
receiver_locations = [10, -0.6]
frequency_peak = model["acquisition"]["frequency"]
final_time = model["timeaxis"]["tf"]
dt = model["timeaxis"]["dt"]


def ricker_wavelet(t, fs, amp=1.0):
    ts = 1.5
    t0 = t - ts * np.sqrt(6.0) / (np.pi * fs)
    return (amp * (1.0 - (1.0 / 2.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0)
            * np.exp((-1.0 / 4.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0))


source_mesh = VertexOnlyMesh(mesh, [source_locations])
source_mesh.coordinates.checkpoint_time_dependent = False
V_s = VectorFunctionSpace(source_mesh, "DG", 0)

d_s = Function(V_s)
spacing = 0.02
d_s.assign(1000)

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
    damp = (sigmax + sigmaz) * inner(((u - u_nm1) / Constant(2.0 * dt)) , v) * dx(scheme=quad_rule)
    F = m + a + damp
    lin_var = LinearVariationalProblem(lhs(F), rhs(F) + source_function, u_np1, constant_jacobian=True)
    solver_parameters = {"mat_type": "matfree", "ksp_type": "preonly", "pc_type": "jacobi"}
    solver = LinearVariationalSolver(lin_var,solver_parameters=solver_parameters)
    return solver, u_np1, u_n, u_nm1

print("Running the receivers")
receiver_mesh = VertexOnlyMesh(mesh, [receiver_locations])
V_r = VectorFunctionSpace(receiver_mesh, "DG", 0)
print("Running the true data")
true_data_receivers = []
total_steps = int(final_time / dt) + 1
f = Cofunction(V.dual())  # Wave equation forcing term.
solver, u_np1, u_n, u_nm1 = wave_equation_solver(rho, vs_true, vp_true, f, dt, V)
interpolate_receivers = interpolate(u_np1, V_r)
output_file = VTKFile("outputs/true_data_elastic_solid.pvd")
start = time.time()
for step in range(total_steps):
    f.assign(ricker_wavelet(step * dt, frequency_peak) * q_s)
    solver.solve()
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    true_data_receivers.append(assemble(interpolate_receivers).dat.data)
    if step % 500 == 0:
        print("Writing step %d" % step, flush=True)
        output_file.write(u_np1)
    if norm(u_np1) > 1e10:
        raise ValueError("The simulation has diverged.")
        break

print("Time taken", time.time() - start)

np.save(file_name, true_data_receivers)
