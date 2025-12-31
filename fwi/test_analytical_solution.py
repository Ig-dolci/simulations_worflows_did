import numpy as np
from scipy.special import hankel2
import matplotlib.pyplot as plt
import finat
import FIAT
from firedrake import *
from firedrake.__future__ import interpolate
# Hide warnings
import warnings
warnings.filterwarnings("ignore")


model = {}

frequency = 90.0
offset = 0.5
c_value = 1.6


model["mesh"] = {
    "xmin": 0.0,
    "xmax": 0.4,
    "zmax": 0.4,
    "zmin": 0.0,
}

model["acquisition"] = {
        "delay_type": "time",
        "frequency": frequency,
        "delay": c_value / frequency,
        "source_locations": [(0.2, 0.2)],
        "receiver_locations": [(0.26, 0.26)],
}

model["timeaxis"] = {
    "t0": 0.0,  # Initial time for event
    "ft": 0.15,  # Final time for event
    "dt": 0.0001,  # time step
}

number_of_elements = 40
L = model["mesh"]["xmax"] - model["mesh"]["xmin"]


def ricker(f, T, dt, t0):
    t = np.linspace(-t0, T-t0, int(T/dt))
    tt = (np.pi**2) * (f**2) * (t**2)
    y = (1.0 - 2.0 * tt) * np.exp(- tt)
    return y


# def analytical_solution(model, c_value, offset):
#     total_steps = int(model["timeaxis"]["ft"] / model["timeaxis"]["dt"]) + 1
#     # Constantes de Fourier
#     nf = int(total_steps / 2 + 1)
#     final_time = model["timeaxis"]["ft"]
#     frequency_axis = (1.0 / final_time) * np.arange(nf)
#     full_ricker_wavelet = []
#     for i in range(total_steps):
#         full_ricker_wavelet.append(ricker_wavelet(i * model["timeaxis"]["dt"], frequency))
#     full_ricker_wavelet = np.array(full_ricker_wavelet)
#     # FOurier tranform of ricker wavelet
#     fft_rw = np.fft.fft(full_ricker_wavelet)
#     fft_rw = fft_rw[0:nf]

#     U_a = np.zeros((nf), dtype=complex)
#     for a in range(1, nf - 1):
#         k = 2 * np.pi * frequency_axis[a] / c_value
#         tmp = k * offset
#         U_a[a] = -1j * np.pi * hankel2(0.0, tmp) * fft_rw[a]

#     U_t = 1.0 / (2.0 * np.pi) * np.real(np.fft.ifft(U_a[:], total_steps))

#     return np.real(U_t)

def analytical(model, c_value, h):
    dt = model["timeaxis"]["dt"]
    nt = int(model["timeaxis"]["ft"] / dt)
    # Fourier constants
    nf = int(nt/2 + 1)
    fnyq = 1. / (2 * dt)
    df = 1.0 / model["timeaxis"]["ft"]
    faxis = df * np.arange(nf)
    f0 = model["acquisition"]["frequency"]
    full_wavelet = ricker(f0, model["timeaxis"]["ft"], dt, 1.5/f0)
    # for i in range(nt):
    #     full_wavelet[i] = ricker_wavelet(i * model["timeaxis"]["dt"], f0)

    # Take the Fourier transform of the source time-function
    R = np.fft.fft(full_wavelet)
    R = R[0:nf]
    nf = len(R)

    # Compute the Hankel function and multiply by the source spectrum
    U_a = np.zeros((nf), dtype=complex)
    sx = model["acquisition"]["source_locations"][0][0]
    sz = model["acquisition"]["source_locations"][0][1]
    rx = model["acquisition"]["receiver_locations"][0][0]
    rz = model["acquisition"]["receiver_locations"][0][1]
    for a in range(1, nf-1):
        k = 2 * np.pi * faxis[a] / c_value
        tmp = k * np.sqrt(((rx - sx))**2 + ((rz - sz))**2)
        U_a[a] = -1j * np.pi * hankel2(0.0, tmp) * R[a]

    # Do inverse fft on 0:dt:T and you have analytical solution
    U_t = 1.0/(2.0 * np.pi) * np.real(np.fft.ifft(U_a[:], nt))

    # The analytic solution needs be scaled by dx^2 to convert to pressure
    # Domain area
    return np.real(U_t) * 1000
    # * h * h * 1000 ** 2


M = 1
my_ensemble = Ensemble(COMM_WORLD, M)
num_sources = my_ensemble.ensemble_comm.size
source_number = my_ensemble.ensemble_comm.rank
# mesh = UnitSquareMesh(100, 100, comm=my_ensemble.comm)
mesh = RectangleMesh(
    number_of_elements, number_of_elements, L, L, quadrilateral=True)

A = L * L
d = assemble(CellDiameter(mesh)*dx) / A
# Compute the cell facets of a square mesh
h = ((d * d)/2)**0.5
u = analytical(model, c_value, h)


source_locations = model["acquisition"]["source_locations"]
receiver_locations = model["acquisition"]["receiver_locations"]
dt = model["timeaxis"]["dt"]  # time step in seconds
final_time = model["timeaxis"]["ft"]  # final time in seconds
frequency_peak = model["acquisition"]["frequency"]  # The dominant frequency of the Ricker wavelet in Hz.


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

element = FiniteElement('CG', mesh.ufl_cell(), degree=4, variant='spectral')
quad_rule = gauss_lobatto_legendre_cube_rule(dimension=2, degree=4)
V = FunctionSpace(mesh, element)
x, z = SpatialCoordinate(mesh)
c_true = Function(V).interpolate(c_value)


def ricker_wavelet(t, fs, amp=1.0):
    ts = 1.5
    t0 = t - ts * np.sqrt(6.0) / (np.pi * fs)
    return (amp * (1.0 - (1.0 / 2.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0)
            * np.exp((-1.0 / 4.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0))


source_mesh = VertexOnlyMesh(mesh, source_locations)
V_s = FunctionSpace(source_mesh, "DG", 0)
d_s = Function(V_s)
d_s.interpolate(c_value * c_value * 1000)
                # * h * h * 1000**2)
source_cofunction = assemble(d_s * TestFunction(V_s) * dx)
q_s = Cofunction(V.dual()).interpolate(source_cofunction)


def wave_equation_solver(c, source_function, dt, V):
    u = TrialFunction(V)
    v = TestFunction(V)
    u_np1 = Function(V, name="u_np1")  # timestep n+1
    u_n = Function(V, name="u_n")  # timestep n
    u_nm1 = Function(V, name="u_nm1")  # timestep n-1
    time_term = (u - 2.0 * u_n + u_nm1) / Constant(dt**2) * v * dx(scheme=quad_rule)
    a = c * c * dot(grad(u_n), grad(v)) * dx(scheme=quad_rule)
    F = time_term + a
    lin_var = LinearVariationalProblem(
        lhs(F), rhs(F) + source_function, u_np1, constant_jacobian=True)
    solver_parameters = {"mat_type": "matfree", "ksp_type": "preonly", "pc_type": "jacobi"}
    solver = LinearVariationalSolver(lin_var, solver_parameters=solver_parameters)
    return solver, u_np1, u_n, u_nm1


receiver_mesh = VertexOnlyMesh(mesh, receiver_locations)
receiver_mesh.coordinates.checkpoint_time_dependent = False
V_r = FunctionSpace(receiver_mesh, "DG", 0)
true_data_receivers = []
total_steps = int(final_time / dt)
f = Cofunction(V.dual())  # Wave equation forcing term.
solver, u_np1, u_n, u_nm1 = wave_equation_solver(c_true, f, dt, V)
interpolate_receivers = interpolate(u_np1, V_r)
output_file = VTKFile("outputs/true_data_3.pvd")
full_ricker_wavelet = ricker(frequency_peak, final_time, dt, 1.5/frequency_peak)

for step in range(total_steps):
    f.assign(full_ricker_wavelet[step] * q_s)
    solver.solve()
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    # solution.append(u_n.copy(deepcopy=True))
    true_data_receivers.append(assemble(interpolate_receivers).dat.data)
    if step % 100 == 0:
        print("Writing step %d" % step, flush=True)
        output_file.write(u_np1)
    if norm(u_np1) > 1e10:
        raise ValueError("The simulation has diverged.")
        break


VTKFile("outputs/true_data.pvd").write(u_np1)

# Plot the analytical and numerical solutions
plt.plot(u, label="Analytical Solution")
plt.plot(true_data_receivers, label="Numerical Solution")
plt.title("Comparison of Analytical and Numerical Solutions")
plt.legend()
plt.grid()
plt.xlabel("Time steps")
plt.ylabel("Pressure (Pa)")
plt.show()


