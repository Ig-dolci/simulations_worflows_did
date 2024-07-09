# import firedrake as fd
from firedrake import *
from firedrake.adjoint import *
import math
import finat
continue_annotation()
mesh = UnitSquareMesh(50, 50)

V = FunctionSpace(mesh, "KMV", 2)

u = TrialFunction(V)
v = TestFunction(V)

u_np1 = Function(V)  # timestep n+1
u_n = Function(V)    # timestep n
u_nm1 = Function(V)  # timestep n-1

outfile = VTKFile("out.pvd")

T = 1.0
dt = 0.001
t = 0
step = 0

freq = 6
c = Constant(1.5)

def RickerWavelet(t, freq, amp=1.0):
    # Shift in time so the entire wavelet is injected
    t = t - (math.sqrt(6.0) / (math.pi * freq))
    return amp * (
        1.0 - (1.0 / 2.0) * (2.0 * math.pi * freq) * (2.0 * math.pi * freq) * t * t
    )

def delta_expr(x0, x, y, sigma_x=2000.0):
    sigma_x = Constant(sigma_x)
    return exp(-sigma_x * ((x - x0[0]) ** 2 + (y - x0[1]) ** 2))

quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree(), "KMV")

dxlump=dx(scheme=quad_rule)

m = (u - 2.0 * u_n + u_nm1) / Constant(dt * dt) * v * dxlump

a = c*c*dot(grad(u_n), grad(v)) * dx

x, y = SpatialCoordinate(mesh)
source = Constant([0.5, 0.5])
ricker = Constant(0.0)
ricker.assign(RickerWavelet(t, freq))

R = Cofunction(V.dual())

F = m + a -  delta_expr(source, x, y)*ricker * v * dx
a, r = lhs(F), rhs(F)
A = assemble(a)
solver = LinearSolver(A, solver_parameters={"ksp_type": "preonly", "pc_type": "jacobi"})


step = 0
while t < T:
    step += 1

    # Update the RHS vector according to the current simulation time `t`

    ricker.assign(RickerWavelet(t, freq))

    R = assemble(r, tensor=R)

    # Call the solver object to do point-wise division to solve the system.

    solver.solve(u_np1, R)

    # Exchange the solution at the two time-stepping levels.

    u_nm1.assign(u_n)
    u_n.assign(u_np1)

    # Increment the time and write the solution to the file for visualization in ParaView.

    t += dt
    if step % 10 == 0:
        print("Elapsed time is: "+str(t))
        outfile.write(u_n, time=t)

J = assemble(0.5 * (u_n - u_n) ** 2 * dx)

J_hat = ReducedFunctional(J, Control(c))