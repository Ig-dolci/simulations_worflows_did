from firedrake import *
from firedrake.adjoint import *
continue_annotation()
from firedrake.ml.jax import *
from jax import config
config.update('jax_enable_x64', True)

tape=get_working_tape()

class Model():
    def __init__(self):
        pass
    
    def __call__(self, u):
        u = u.at[0].set(2 * u[1])
        return u

mesh = UnitSquareMesh(2, 2)
V = FunctionSpace(mesh, "CG", 1)

u0 = Function(V)
u0.vector().set_local(1)

model = Model()
N = ml_operator(model, function_space=V, inputs_format=1)
u1 = assemble(N(u0))

J = assemble(u1*dx)

# tape.visualise_dot('test.dot')

dJdu0 = compute_gradient(J, Control(u0))