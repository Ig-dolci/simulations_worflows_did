import finat
from firedrake import *
from firedrake.adjoint import *
from firedrake.__future__ import Interpolator, interpolate
import gc
import numpy as np
from checkpoint_schedules import Revolve, SingleMemoryStorageSchedule
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--annotate", type=str, default="False")
parser.add_argument("--checkpointing", type=str, default="False")
parser.add_argument("--schedule", type=str, default="revolve")
parser.add_argument("--n_checkpoints", type=int, default=10)
parser.add_argument("--function", type=str, default="interpolate")
args = parser.parse_args()
if args.annotate == "True":
    continue_annotation()

tape = get_working_tape()
total_steps = 1000
if args.checkpointing == "True" and args.schedule == "revolve":
    steps = args.n_checkpoints
    tape.enable_checkpointing(Revolve(total_steps, steps))
elif args.checkpointing == "True" and args.schedule == "single":
    print("single")
    tape.enable_checkpointing(SingleMemoryStorageSchedule())

num_sources = 1
source_number = 0
Lx, Lz = 1.0, 1.0
mesh = UnitSquareMesh(400, 400)
V = FunctionSpace(mesh, "CG", 1)


def test_memory_interpolate():
    u_np1 = Function(V)
    u_np1.assign(1.0)
    J = 0.0
    if tape._checkpoint_manager:
        for step in tape.timestepper(iter(range(total_steps))):
            # print("step = ", step, "revolve ")
            u_np1.interpolate(0.0001*(step + 2) * u_np1)
    else:
        for step in range(total_steps):
            # print("step = ", step, "no revolve")
            u_np1.interpolate(0.0001*(step + 2) * u_np1)
    J = assemble(u_np1*u_np1*dx)
    J_hat = ReducedFunctional(J, Control(u_np1))
    J_hat(Function(V).assign(1.0))

#
 
def test_memory_assign():
    u_np1 = Function(V)
    u_np1.assign(1.0)
    J = 0.0
    if tape._checkpoint_manager:
        for step in tape.timestepper(iter(range(total_steps))):
            # print("step = ", step, "revolve ")
            u_np1.assign(0.0001*(step + 2) * u_np1)
    else:
        for step in range(total_steps):
            # print("step = ", step, "no revolve")
            u_np1.assign(0.0001*(step + 2) * u_np1)
    J = assemble(u_np1*u_np1*dx)
    J_hat = ReducedFunctional(J, Control(u_np1))
    J_hat(Function(V).assign(1.0))


def test_memory_solve():
    u_n = Function(V)
    u_n.assign(1.0)
    u = TrialFunction(V)
    v = TestFunction(V)
    dt = Constant(0.00001)
    F = (u - u_n) * v * dx + dt * Constant(1.0) * v * dx
    J = 0.0
    u_np1 = Function(V)
    problem = LinearVariationalProblem(lhs(F), rhs(F), u_np1)
    solver = LinearVariationalSolver(problem)
    if tape._checkpoint_manager:
        for step in tape.timestepper(iter(range(total_steps))):
            # print("step = ", step, "revolve ")
            solver.solve()
            u_n.assign(u_np1)
    else:
        for step in range(total_steps):
            # print("step = ", step, "no revolve")
            solver.solve()
            u_n.assign(u_np1)


def Dt(u, u_, timestep):
    return (u - u_)/timestep


def test_memory_burger():
    u_ = Function(V)
    u = Function(V)
    v = TestFunction(V)
    timestep = Constant(0.0000001)
    x,_ = SpatialCoordinate(mesh)
    ic = project(sin(2.*pi*x), V)
    u_.assign(ic)
    nu = Constant(0.001)
    F = (Dt(u, u_, timestep)*v + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")
    t = 0.0
    problem = NonlinearVariationalProblem(F, u, bcs=bc)
    solver = NonlinearVariationalSolver(problem)
    t += float(timestep)
  
    if tape._checkpoint_manager:
        for t in tape.timestepper(iter(range(total_steps))):
            # print("step = ", t, "revolve ")
            solver.solve()
            u_.assign(u)
    else:
        for t in range(total_steps):
            # print("step = ", t, "no revolve")
            solver.solve()
            u_.assign(u)
    J = assemble(u*u*dx)
    J_hat = ReducedFunctional(J, Control(ic))
    J_hat(ic)


if args.function == "interpolate":
    test_memory_interpolate()
elif args.function == "assign":
    test_memory_assign()
elif args.function == "burger":
    test_memory_burger()