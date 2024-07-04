import numpy as np
from scipy.special import hankel2
import matplotlib.pyplot as plt

# Source and receiver geometries
f0 = .09
c0 = 1.5
dt = 0.1
Nx = 800
Ny = 800
spatial_dx = 0.5
source_locations = [(200, 200.0)]
receiver_locations = [(260, 260)]


# Source and receiver geometries
src_coordinates = np.empty((1, 2))
src_coordinates[0, :] = 200.

# Single receiver offset 100 m from source
rec_coordinates = np.empty((1, 2))
rec_coordinates[:, :] = 260.

# Source and receiver coordinates
sx, sz = src_coordinates[0, :]
rx, rz = rec_coordinates[0, :]


# Define a Ricker wavelet shifted to zero lag for the Fourier transform
def ricker(f, T, dt, t0):
    t = np.linspace(-t0, T-t0, int(T/dt))
    tt = (np.pi**2) * (f**2) * (t**2)
    y = (1.0 - 2.0 * tt) * np.exp(- tt)
    return y


def ricker_wavelet(t, f):
    a = 2 * (np.pi * f)**2
    return (1 - a * t**2) * np.exp(-a * t**2 / 2)

def analytical(nt, time):
    # Fourier constants
    nf = int(nt/2 + 1)
    df = 1.0 / time[-1]
    faxis = df * np.arange(nf)
    
    wavelet = ricker(f0, time[-1], dt, 1.5/f0)

    # Take the Fourier transform of the source time-function
    R = np.fft.fft(wavelet)
    R = R[0:nf]
    nf = len(R)

    # Compute the Hankel function and multiply by the source spectrum
    U_a = np.zeros((nf), dtype=complex)
    for a in range(1, nf-1):
        k = 2 * np.pi * faxis[a] / c0
        tmp = k * np.sqrt(((rx - sx))**2 + ((rz - sz))**2)
        U_a[a] = -1j * np.pi * hankel2(0.0, tmp) * R[a]

    # Do inverse fft on 0:dt:T and you have analytical solution
    U_t = 1.0/(2.0 * np.pi) * np.real(np.fft.ifft(U_a[:], nt))
    
    # The analytic solution needs be scaled
    return np.real(U_t) * (spatial_dx**2)


# Number of time steps
nt = 1501
t0 = 0.
tn = dt * nt
time1 = np.linspace(0.0, 3000., 30001)
time = np.linspace(t0, tn, nt)
U_t = analytical(30001, time1)
U_t = U_t[0:1501]

import finat
from firedrake import *
from firedrake.__future__ import Interpolator, interpolate

def wave_equation_solver(c, source_function, dt, V):
    u = TrialFunction(V)
    v = TestFunction(V)
    u_np1 = Function(V) # timestep n+1
    u_n = Function(V) # timestep n
    u_nm1 = Function(V) # timestep n-1
    # Quadrature rule for lumped mass matrix.
    quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree(), "KMV")
    time_term = (1 / (c * c)) * (u - 2.0 * u_n + u_nm1) / Constant(dt**2) * v * dx(scheme=quad_rule)
    nf = (1 / c) * ((u_n - u_nm1) / dt) * v * ds
    a = dot(grad(u_n), grad(v)) * dx(scheme=quad_rule)
    F = time_term + a + nf
    lin_var = LinearVariationalProblem(lhs(F), rhs(F) + source_function, u_np1)
    solver_parameters = {"mat_type": "matfree", "ksp_type": "preonly", "pc_type": "jacobi"}
    solver = LinearVariationalSolver(lin_var,solver_parameters=solver_parameters)
    return solver, u_np1, u_n, u_nm1


mesh = SquareMesh(Nx, Ny, 400, 400)
V = FunctionSpace(mesh, "KMV", 1)

x, z = SpatialCoordinate(mesh)
c_true = Function(V).assign(c0)


receiver_mesh = VertexOnlyMesh(mesh, receiver_locations)
V_r = FunctionSpace(receiver_mesh, "DG", 0)

source_mesh = VertexOnlyMesh(mesh, source_locations)
V_s = FunctionSpace(source_mesh, "DG", 0)

d_s = Function(V_s)
d_s.assign(1.0)
source_cofunction = assemble(d_s * TestFunction(V_s) * dx)
q_s = Cofunction(V.dual()).interpolate(source_cofunction)

true_data_receivers = []
f = Cofunction(V.dual())  # Wave equation forcing term.
solver, u_np1, u_n, u_nm1 = wave_equation_solver(c_true, f, dt, V)
interpolate_receivers = Interpolator(u_np1, V_r).interpolate()
frequency_peak = f0
final_time = tn
wavelet = ricker(frequency_peak, final_time, dt, 1.5/frequency_peak)
for step, time_step in enumerate(time[0:-2]):
    print("Step: ", time_step, " of ", time[-1])
    print("Wavelet: ", wavelet[step])
    f.assign(wavelet[step] * q_s)
    solver.solve()
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    true_data_receivers.append(assemble(interpolate_receivers).dat.data[0] * spatial_dx**2)


VTKFile("wavefieldperf.pvd").write(u_np1)

# plt.figure(figsize=(8,8))
# # amax = np.max(np.abs(ref_u.data[1,:,:]))
# # plt.imshow(ref_u.data[1,:,:], vmin=-1.0 * amax, vmax=+1.0 * amax, cmap="seismic")
# plt.plot(2*sx+40, 2*sz+40, 'r*', markersize=11, label='source')   # plot position of the source in model, add nbl for correct position
# plt.plot(2*rx+40, 2*rz+40, 'k^', markersize=8, label='receiver')  # plot position of the receiver in model, add nbl for correct position
# plt.legend()
# plt.xlabel('x position (m)')
# plt.ylabel('z position (m)')
# plt.savefig('wavefieldperf.pdf')
print(max(true_data_receivers)/max(U_t[:]))
# Plot trace
plt.figure(figsize=(12,8))
plt.subplot(1,1,1)
plt.plot(time[0:-2], true_data_receivers, '-b', label='numerical')
plt.plot(time, U_t[:], '--r', label='analytical')
# plt.xlim([0,150])
# plt.ylim([1.15*np.min(U_t[:]), 1.15*np.max(U_t[:])])
plt.xlabel('time (ms)')
plt.ylabel('amplitude')
plt.legend()
# plt.subplot(2,1,2)
# plt.plot(time, 100 *(ref_rec.data[:, 0] - U_t[:]), '-b', label='difference x100')
# plt.xlim([0,150])
# plt.ylim([1.15*np.min(U_t[:]), 1.15*np.max(U_t[:])])
# plt.xlabel('time (ms)')
# plt.ylabel('amplitude x100')
# plt.legend()
# plt.savefig('ref.pdf')
plt.show()
