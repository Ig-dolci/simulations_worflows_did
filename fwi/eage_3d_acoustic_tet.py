# /from tools import model_interpolate
from tools import model_eage_salt as model
from tools import parameter_interpolate, read_raw_binary_3d
import zipfile
from firedrake import *
import finat
import FIAT
import numpy as np
import time
from mpi4py import MPI
import argparse
# 3D mesh cases (elem_x x elem_y x elem_z) - refined for error < 0.1
# case 1 - 28x28x14 mesh
# case 2 - 36x36x18 mesh
# case 3 - 43x43x21 mesh
# case 4 - 51x51x25 mesh
# case 5 - 57x57x28 mesh
# case 6 - 64x64x32 mesh
# case 7 - 71x71x35 mesh
# case 8 - 79x79x39 mesh
# case 9 - 85x85x42 mesh
# case 10 - 93x93x46 mesh
# case 11 - 99x99x49 mesh
# case 12 - 107x107x53 mesh
# case 13 - 113x113x56 mesh
# reference case - 129x129x64 mesh
M = 1

# Parse command line arguments for 3D quad mesh
parser = argparse.ArgumentParser(description='Run acoustic wave simulation with different mesh sizes')
parser.add_argument('--case', type=str, default='1', help='Case number or "ref" for reference')
parser.add_argument('--elem-x', type=int, default=28, help='Number of elements in x direction')
parser.add_argument('--elem-y', type=int, default=28, help='Number of elements in y direction')
parser.add_argument('--elem-z', type=int, default=14, help='Number of elements in z direction')
args = parser.parse_args()

# Use arguments from command line
elem_x = args.elem_x
elem_y = args.elem_y
elem_z = args.elem_z
file_name = f"comp_performance/rec_data_3d_tet_acoustic_{args.case}.npy"

print(f"Running case {args.case}: {elem_x}x{elem_y}x{elem_z} mesh")
print(f"Output file: {file_name}")

num_sources = 1
source_number = 0
lx = 13520.0
ly = 13520.0
lz = 4200.0  # Positive value - BoxMesh creates from 0 to lz
mesh = BoxMesh(elem_x, elem_y, elem_z, lx, ly, lz, hexahedral=False)  # Tetrahedral mesh for KMV

# Transform Z coordinates from [0, 4200] to [-4200, 0]
mesh.coordinates.dat.data[:, 2] = -mesh.coordinates.dat.data[:, 2]


degree = 3  # KMV degree
element = FiniteElement('KMV', mesh.ufl_cell(), degree=degree)

V = FunctionSpace(mesh, element)
quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree(), "KMV")
print("DOFs", V.dim())
comm = MPI.COMM_WORLD
path = "/Users/ddolci/work/my_runs/spyro_meshes/Salt_Model_3D/3-D_Salt_Model/VEL_GRIDS/"
if comm.rank == 0:
    # Extract binary file Saltf@@ from SALTF.ZIP
    zipfile.ZipFile(path + "SALTF.ZIP", "r").extract("Saltf@@", path=path)
print("Reading velocity model from binary file...")
fname = model["mesh"]["vp"]
# EAGE Salt model: n1=676, n2=676, n3=210 (from salt.h header)
# SEP format uses big-endian byte order
vp = read_raw_binary_3d(
    fname,
    nx=model["mesh"]["nx"],
    ny=model["mesh"]["ny"],
    nz=model["mesh"]["nz"],
    dtype='>f4'
)
# Grid spacing is 20m
# For 3D, disable BCs padding (parameter_interpolate needs 3D PML support)
model_no_bc = model.copy()
model_no_bc["BCs"] = model["BCs"].copy()
model_no_bc["BCs"]["status"] = False
vp_true = parameter_interpolate(model_no_bc, V, vp, l_grid=20.0, name="vp_true")
VTKFile("outputs/vp_true.pvd").write(vp_true)

# 3D source and receiver locations (x, y, z)
source_locations = [6760.0, 6760.0, -100.0]  # Center of domain, 100m depth
receiver_locations = [10000.0, 6760.0, -100.0]  # Along x-axis


def ricker_wavelet(t, fs, amp=1.0):
    ts = 0.0
    t0 = t - ts * np.sqrt(6.0) / (np.pi * fs)
    return (amp * (1.0 - (1.0 / 2.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0)
            * np.exp((-1.0 / 4.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0))


source_mesh = VertexOnlyMesh(mesh, [source_locations])
V_s = FunctionSpace(source_mesh, "DG", 0)


# 3D source scaling - reduced amplitude to prevent divergence
# Using smaller scaling factor for stability
val = 1e-6 * (8.0 / elem_x) ** 2 * (8.0 / elem_y) ** 2 * (8.0 / elem_z) ** 2
source_cofunction = assemble(TestFunction(V_s) * dx)
q_s = Cofunction(V.dual()).interpolate(source_cofunction)


def acoustic_wave_equation_3d(vp, dt, V, f):
    """Setup 3D acoustic wave equation solver."""
    u = TrialFunction(V)
    v = TestFunction(V)
    u_np1 = Function(V, name="u_np1")  # timestep n+1
    u_n = Function(V, name="u_n")  # timestep n
    u_nm1 = Function(V, name="u_nm1")  # timestep n-1
    
    # Wave equation: d²u/dt² = c² ∇²u
    # Weak form: M(u-2u_n+u_nm1)/dt² - K u_n = f
    # Note: vp is in km/s, mesh coordinates are in meters, vp*vp gives (km/s)²
    mass_term = ((u - 2.0 * u_n + u_nm1) / Constant(dt**2) * v * 
                 dx(scheme=quad_rule))
    stiffness_term = (vp * vp * dot(grad(u_n), grad(v)) * 
                      dx(scheme=quad_rule))
    F = mass_term + stiffness_term
    
    lin_var = LinearVariationalProblem(
        lhs(F), rhs(F) + f, u_np1, constant_jacobian=True
    )
    solver_parameters = {
        "mat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "jacobi"
    }
    solver = LinearVariationalSolver(lin_var, solver_parameters=solver_parameters)
    return solver, u_np1, u_n, u_nm1


# Time parameters
final_time = model["timeaxis"]["tf"]
frequency_peak = model["acquisition"]["frequency"]

dt_model = model["timeaxis"]["dt"]

print(f"Max velocity: {vp_true.dat.data.max():.2f} Km/s")
# Always use CFL-stable dt with safety factor
dt = dt_model
print(f"Using dt = {dt:.6f} s ({dt*1000:.3f} ms) (0.5 * CFL)")

receiver_mesh = VertexOnlyMesh(mesh, [receiver_locations])
V_r = FunctionSpace(receiver_mesh, "DG", 0)
print("Running 3D acoustic wave simulation...")
true_data_receivers = []
total_steps = int(final_time / dt) + 1
f = Cofunction(V.dual())  # Wave equation forcing term
solver, u_np1, u_n, u_nm1 = acoustic_wave_equation_3d(
    vp_true, dt, V, f)
interpolate_receivers = interpolate(u_np1, V_r)
output_file = VTKFile("outputs/true_data_3D.pvd")
start = time.perf_counter()

for step in range(total_steps):
    f.assign(ricker_wavelet(step * dt, frequency_peak) * q_s)
    solver.solve()
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    true_data_receivers.append(assemble(interpolate_receivers).dat.data)
    
    u_norm = norm(u_np1)
    if step % 10000 == 0:
        print(f"Writing step {step}/{total_steps}, norm={u_norm:.2e}", flush=True)
        output_file.write(u_np1)
    if u_norm > 1e10 or np.isnan(u_norm):
        print(f"ERROR: The simulation has diverged at step {step}.")
        print(f"  u_np1 norm: {u_norm:.2e}")
        print(f"  u_np1 min/max: {u_np1.dat.data.min():.2e} / {u_np1.dat.data.max():.2e}")
        break

end = time.perf_counter()

# Gather results from all processes to rank 0
if comm.rank == 0:
    print("Time taken", end - start)
    print(f"Gathering results from {comm.size} processes...")

# Convert list to numpy array for gathering
true_data_receivers_array = np.array(true_data_receivers)

# Gather all receiver data to rank 0
gathered_data = comm.gather(true_data_receivers_array, root=0)
if comm.rank == 0:
    print(f"Checking the gathered data: {gathered_data[-1]}")
# Save results only on rank 0
if comm.rank == 0:
    # Find which rank has the actual receiver data (non-empty array)
    # The receiver might only be in one rank's partition
    valid_data = None
    for rank_data in gathered_data:
        if rank_data.size > 0 and rank_data.shape[1] > 0:
            valid_data = rank_data
            break
    
    if valid_data is not None:
        print(f"Saving results to {file_name}")
        print(f"Data shape: {valid_data.shape}")
        np.save(file_name, valid_data)
        print(f"✓ Results saved successfully to {file_name}")
    else:
        print("Warning: No valid receiver data found in any rank!")
        # Save empty array to avoid missing file errors
        np.save(file_name, np.array([]))