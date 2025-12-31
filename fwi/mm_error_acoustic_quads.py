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
import argparse
# read a hdf5 file
M = 1

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run acoustic wave simulation with different mesh sizes')
parser.add_argument('--case', type=str, default='13', help='Case number or "ref" for reference')
parser.add_argument('--elem-x', type=int, default=520, help='Number of elements in x direction')
parser.add_argument('--elem-y', type=int, default=260, help='Number of elements in y direction')
args = parser.parse_args()
# case 1 - 80x40 mesh
# case 2 - 120x60 mesh
# case 3 - 160x80 mesh
# case 4 - 200x100 mesh
# case 5 - 240x120 mesh
# case 6 - 280x140 mesh
# case 7 - 320x160 mesh
# case 8 - 360x180 mesh
# case 9 - 400x200 mesh
# case 10 - 440x220 mesh
# case 11 - 480x240 mesh
# case 12 - 500x250 mesh
# case 13 - 520x260 mesh
# reference case - 540x270 mesh

# Use arguments from command line
elem_x = args.elem_x
elem_y = args.elem_y
file_name = f"comp_performance/rec_data_quad_no_adaptive_acoustic_{args.case}.npy"

print(f"Running case {args.case}: {elem_x}x{elem_y} mesh")
print(f"Output file: {file_name}")

my_ensemble = Ensemble(MPI.COMM_WORLD, M)
num_sources = my_ensemble.ensemble_comm.size
source_number = my_ensemble.ensemble_comm.rank
# mesh = RectangleMesh(100, 100, 1., 1., comm=my_ensemble.comm, quadrilateral=True)
mesh = RectangleMesh(elem_x, elem_y,
                     model["mesh"]["xmax"] - model["mesh"]["xmin"] + 2*model["BCs"]["lx"],
                     model["mesh"]["zmin"] - model["mesh"]["zmax"] - model["BCs"]["lz"],
                     quadrilateral=True, distribution_parameters={"partitioner_types": "ptscotch"})

for i in range(len(mesh.coordinates.dat.data)):
    mesh.coordinates.dat.data[i, 0] += model["mesh"]["xmin"] - model["BCs"]["lx"]

mesh.coordinates.dat.data[:, 1] += model["mesh"]["zmax"]

dt = model["timeaxis"]["dt"]  # time step in seconds
final_time = model["timeaxis"]["tf"]  # final time in seconds
# The dominant frequency of the Ricker wavelet in KHz.
frequency_peak = model["acquisition"]["frequency"]


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

V = FunctionSpace(mesh, element)
print("DOFs", V.dim())

vp = read_segy(model["mesh"]["vp"])
vp_true = parameter_interpolate(model, V, vp.T, l_grid=1.25, name="vp_true")
VTKFile("outputs/vp_true.pvd").write(vp_true)
print(" max vp_true ", vp_true.dat.data.max())
quit()
source_locations = [9 * 1000, -0.1 * 1000]
receiver_locations = [10 * 1000, -0.1 * 1000]

def ricker_wavelet(t, fs, amp=1.0):
    ts = 0.0
    t0 = t- ts * np.sqrt(6.0) / (np.pi * fs)
    return (amp * (1.0 - (1.0 / 2.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0)
            * np.exp((-1.0 / 4.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0))

source_mesh = VertexOnlyMesh(mesh, [source_locations])
V_s = FunctionSpace(source_mesh, "DG", 0)

val = 1. * (8 / elem_x) * (8 / elem_x)
source_cofunction = assemble(TestFunction(V_s) * dx)
q_s = Cofunction(V.dual()).interpolate(source_cofunction)


def sh_wave_equation(vp, dt, V, f, quad_rule):

    u = TrialFunction(V)
    v = TestFunction(V)
    u_np1 = Function(V, name="u_np1") # timestep n+1
    u_n = Function(V, name="u_n") # timestep n
    u_nm1 = Function(V, name="u_nm1") # timestep n-1
    time_term = (u - 2.0 * u_n + u_nm1) / Constant(dt**2) * v * dx(scheme=quad_rule)
    a = vp*vp * dot(grad(u_n), grad(v)) * dx(scheme=quad_rule)
    F = time_term + a
    lin_var = LinearVariationalProblem(lhs(F), rhs(F) + f, u_np1, constant_jacobian=True)
    solver_parameters = {"mat_type": "matfree", "ksp_type": "preonly", "pc_type": "jacobi"}
    solver = LinearVariationalSolver(lin_var, solver_parameters=solver_parameters)
    return solver, u_np1, u_n, u_nm1


receiver_mesh = VertexOnlyMesh(mesh, [receiver_locations])
V_r = FunctionSpace(receiver_mesh, "DG", 0)
print("Running the true data")
true_data_receivers = []
total_steps = int(final_time / dt) + 1
f = Cofunction(V.dual())  # Wave equation forcing term.
solver, u_np1, u_n, u_nm1 = sh_wave_equation(vp_true, dt, V, f, quad_rule)
interpolate_receivers = interpolate(u_np1, V_r)
output_file = VTKFile("outputs/true_data_2D.pvd")
start = time.perf_counter()
for step in range(total_steps):
    f.assign(ricker_wavelet(step * dt, frequency_peak)*q_s)
    solver.solve()
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    true_data_receivers.append(assemble(interpolate_receivers).dat.data)
    if step % 100 == 0:
        print("Writing step %d" % step, flush=True)
        output_file.write(u_np1)
    if norm(u_np1) > 1e10:
        raise ValueError("The simulation has diverged.")
        break

end = time.perf_counter()
print("Time taken", end - start)
print(f"Saving results to {file_name}")

np.save(
    file_name,
    true_data_receivers)
    
print(f"✓ Results saved successfully to {file_name}")