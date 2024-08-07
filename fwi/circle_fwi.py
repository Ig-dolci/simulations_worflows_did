# Full-waveform inversion: spatial and wave sources parallelism
# =================================================================
# This tutorial demonstrates using Firedrake to solve a Full-Waveform Inversion problem employing
# gradient computation using the algorithmic differentiation, pyadjoint. We also show how easy it is to
# configure the spatial and wave sources parallelism in order to compute the cost functions and their
# gradient on this optimisation problem.
#
# .. rst-class:: emphasis
#
#     This tutorial was prepared by `Daiane I. Dolci <mailto:d.dolci@imperial.ac.uk>`__ 
#     and Jack Betteridge.
#
# Full-waveform inversion (FWI) consists of a local optimisation, where the goal is to minimise
# the misfit between observed and predicted seismogram data. The misfit is quantified by a functional,
# which in general is a summation of the cost functions for multiple sources:
#
# .. math::
#
#        J = \sum_{s=1}^{N_s} J_s(u, u^{obs}),  \quad \quad (1)
#
# where :math:`N_s` is the number of sources, and :math:`J_s(u, u^{obs})` is the cost function
# for a single source. Following :cite:`Tarantola:1984`, the cost function for a single
# source can be measured by the :math:`L^2` norm:
#
# .. math::
#    
#     J_s(u, u^{obs}) = \sum_{r=0}^{N_r} \sum_{t=0}^{T} \left(
#         u(c,\mathbf{x}_r,t) - u^{obs}(c, \mathbf{x}_r,t)\right)^2,   \quad \quad (2)
#
# where :math:`u = u(c, \mathbf{x}_r,t)` and :math:`u_{obs} = u_{obs}(c,\mathbf{x}_r,t)`,
# are respectively the computed and observed data, both recorded at a finite number
# of receivers :math:`N_r` located at the point positions :math:`\mathbf{x}_r \in \Omega`,
# in a time-step interval :math:`[0, T]`, where :math:`T` is the total time-step.
#
# The predicted data is here modeled here by an acoustic wave equation,
#
# .. math::
#
#     \frac{\partial^2 u}{\partial t^2}- c^2\frac{\partial^2 u}{\partial \mathbf{x}^2} = f(\mathbf{x},t),  \quad \quad (3)
#
# where :math:`c(\mathbf{x})` is the pressure wave velocity, which is assumed here a piecewise-constant and positive. The
# function :math:`f(\mathbf{x},t)` models a point source function, were the time-dependency is given by the 
# `Ricker wavelet <https://wiki.seg.org/wiki/Dictionary:Ricker_wavelet>`__.
#
# The acoustic wave equation should satisfy the initial conditions :math:`u(\mathbf{x}, 0) = 0 = u_t(\mathbf{x}, 0) = 0`.
# We are employing no-reflective absorbing boundary condition :cite:`Clayton:1977`:
#
# .. math::  \frac{\partial u}{\partial t}- c\frac{\partial u}{\partial \mathbf{x}} = 0, \, \, 
#            \forall \mathbf{x} \, \in \partial \Omega  \quad \quad (4)
#
#
# To solve the wave equation, we consider the following weak form over the domain :math:`\Omega`:
#
# .. math:: \int_{\Omega} \left(
#     \frac{\partial^2 u}{\partial t^2}v + c^2\nabla u \cdot \nabla v\right
#     ) \, dx = \int_{\Omega} f(\mathbf{x},t) v \, dx, \quad \quad (5)
#
# for an arbitrary test function :math:`v\in V`, where :math:`V` is a function space. 
#
#
# In Firedrake, we can simultaneously compute functional values and their gradients for a number multiple sources :math:`N_s`
# using ``Ensemble``. This tutorial demonstrates how an ``Ensemble`` object is employed on the current inversion problem.
# First, we will need to define an ensemble object::

from firedrake import *
import finat
from firedrake.__future__ import interpolate
M = 1
my_ensemble = Ensemble(COMM_WORLD, M)

# ``my_ensemble`` requires a communicator (which by default is ``COMM_WORLD``) and a value ``M``, the "team" size,
# used to configure the ensemble parallelism. Based on the value of ``M`` and the number of MPI processes,
# :class:`~.ensemble.Ensemble` will split the total number of MPI processes in ``COMM_WORLD`` into two
# sub-communicators: ``Ensemble.comm`` the spatial communicator having a unique source that each mesh is
# distributed over and ``Ensemble.ensemble_comm``. ``Ensemble.ensemble_comm`` is used to communicate information
# about the functionals and their gradients computation between different wave sources.
#
# In this case, we want to distribute each mesh over 2 ranks and compute the functional and its gradient
# for 3 wave sources. So we set ``M=2`` and execute this code with 6 MPI ranks. That is: 3 (number of sources) x 2 (M).
# To have a better understanding of the ensemble parallelism, please refer to the
# `Firedrake manual <hhttps://www.firedrakeproject.org/parallelism.html#id8>`__.
#
# The number of sources are set according the source ``my_ensemble.ensemble_comm.size`` (3 in this case)::

num_sources = my_ensemble.ensemble_comm.size

# The source number is defined according to the rank of the ``Ensemble.ensemble_comm``::

source_number = my_ensemble.ensemble_comm.rank

# After configuring ``my_ensemble``, we now consider a two-dimensional square domain with a side length of 1.0 km.
# The mesh is built over the ``my_ensemble.comm`` communicator::
    
Lx, Lz = 1.0, 1.0
mesh = UnitSquareMesh(80, 80, comm=my_ensemble.comm)

# # The basic input for the FWI problem are defined as follows::

import numpy as np
source_locations = np.linspace((0.3, 0.1), (0.7, 0.1), num_sources)
receiver_locations = np.linspace((0.2, 0.9), (0.8, 0.9), 10)
dt = 0.002  # time step
final_time = 0.8  # final time
frequency_peak = 7.0  # The dominant frequency of the Ricker wavelet.

# # We are using a 2D domain, 10 receivers, and 3 sources. Sources and receivers locations are illustrated
# # in the following figure:
# #
# # .. image:: sources_receivers.png
# #     :scale: 70 %
# #     :alt: sources and receivers locations
# #     :align: center
# #
# #        
# # FWI seeks to estimate the pressure wave velocity based on the observed data stored at the receivers.
# # These data are subject to influences of the subsurface medium while waves propagate from the sources.
# # In this example, we emulate observed data by executing the acoustic wave equation with a synthetic
# # pressure wave velocity model. The synthetic pressure wave velocity model is referred to here as the
# # true velocity model (``c_true``). For the sake of simplicity, we consider ``c_true`` consisting of a
# # circle in the centre of the domain, as shown in the coming code cell::

V = FunctionSpace(mesh, "KMV", 1)
x, z = SpatialCoordinate(mesh)
c_true = Function(V).interpolate(2.5 + 1 * tanh(200 * (0.125 - sqrt((x - 0.5) ** 2 + (z - 0.5) ** 2))))

# # .. image:: c_true.png
# #     :scale: 70 %
# #     :alt: true velocity model
# #     :align: center
# #
# #
# # We now seek to model the point source function in weak form, which is the term on the right side of Eq. (5) rewritten
# # as:
# #
# # .. math:: \int_{\Omega} f(\mathbf{x},t) v \, dx = r(t) q_s(\mathbf{x}),  \quad q_s(\mathbf{x}) \in V^{\ast} \quad \quad (6)
# #
# # where :math:`r(t)` is `Ricker wavelet <https://wiki.seg.org/wiki/Dictionary:Ricker_wavelet>`__  coded as follows::

def ricker_wavelet(t, fs, amp=1.0):
    ts = 1.5
    t0 = t - ts * np.sqrt(6.0) / (np.pi * fs)
    return (amp * (1.0 - (1.0 / 2.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0)
            * np.exp((-1.0 / 4.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0))

# # To compute the cofunction :math:`q_s(\mathbf{x})\in V^{\ast}`, we first construct the source mesh over the source location
# # :math:`\mathbf{x}_s`, for the source number ``source_number``::

source_mesh = VertexOnlyMesh(mesh, [source_locations[source_number]])

# Next, we define a function space :math:`V_s` accordingly::

V_s = FunctionSpace(source_mesh, "DG", 0)

# The point source value :math:`d_s(\mathbf{x}_s) = 1.0` is coded as::

d_s = Function(V_s)
d_s.assign(1.0)

# # We then inteporlate a cofunction in :math:`V_s^{\ast}` onto :math:`V^{\ast}` to then have :math:`q_s \in V^{\ast}`::

source_cofunction = assemble(d_s * TestFunction(V_s) * dx)
q_s = Cofunction(V.dual()).interpolate(source_cofunction)


# # The forward wave equation solver is written as follows::



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

# You can find more details about the wave equation with mass lumping on this
# `Firedrake demos <https://www.firedrakeproject.org/demos/higher_order_mass_lumping.py.html>`_.
#
# The receivers mesh and its function space :math:`V_r`::

receiver_mesh = VertexOnlyMesh(mesh, receiver_locations)
V_r = FunctionSpace(receiver_mesh, "DG", 0)

# The receiver mesh is required in order to interpolate the wave equation solution at the receivers.
#
# We are now able to proceed with the synthetic data computations and record them on the receivers::

true_data_receivers = []
total_steps = int(final_time / dt) + 1
f = Cofunction(V.dual())  # Wave equation forcing term.
solver, u_np1, u_n, u_nm1 = wave_equation_solver(c_true, f, dt, V)
interpolate_receivers = interpolate(u_np1, V_r)

for step in range(total_steps):
    f.assign(ricker_wavelet(step * dt, frequency_peak) * q_s)
    solver.solve()
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    true_data_receivers.append(assemble(interpolate_receivers))

# # Next, the FWI problem is executed with the following steps:
# #
# # 1. Set the initial guess for the parameter ``c_guess``;
# #
# # 2. Solve the wave equation with the initial guess velocity model (``c_guess``);
# #
# # 3. Compute the functional :math:`J`;
# #
# # 4. Compute the adjoint-based gradient of :math:`J` with respect to the control parameter ``c_guess``;
# #
# # 5. Update the parameter ``c_guess`` using a gradient-based optimisation method, on this case the L-BFGS-B method;
# #
# # 6. Repeat steps 2-5 until the optimisation stopping criterion is satisfied.
# #
# # **Step 1**: The initial guess is set as a constant field with a value of 1.5 km/s::

c_guess = Function(V).assign(1.5)


# # .. image:: c_initial.png
# #     :scale: 70 %
# #     :alt: initial velocity model
# #     :align: center
# #
# #
# # To have the step 4, we need first to tape the forward problem. That is done by calling::

from firedrake.adjoint import *
from checkpoint_schedules import Revolve
continue_annotation()
tape = get_working_tape()
tape.enable_checkpointing(Revolve(total_steps, 10))
# **Steps 2-3**: Solve the wave equation and compute the functional::

f = Cofunction(V.dual())  # Wave equation forcing term.
solver, u_np1, u_n, u_nm1 = wave_equation_solver(c_guess, f, dt, V)
interpolate_receivers = interpolate(u_np1, V_r)
J_val = 0.0
for step in tape.timestepper(iter(range(total_steps))):
    f.assign(ricker_wavelet(step * dt, frequency_peak) * q_s)
    solver.solve()
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    guess_receiver = assemble(interpolate_receivers)
    misfit = guess_receiver - true_data_receivers[step]
    J_val += 0.5 * assemble(inner(misfit, misfit) * dx)

# # :class:`~.EnsembleReducedFunctional` is employed to recompute in parallel the functional and
# # its gradient associated with the multiple sources (3 in this case)::

J_hat = EnsembleReducedFunctional(J_val, Control(c_guess), my_ensemble)
assert np.allclose(J_hat(c_guess), J_val)
# assert np.allclose(J_hat(c_guess), J_val)
# # The ``J_hat`` object is passed as an argument to the ``minimize`` function (see the Python code below).
# # In the backend, ``J_hat`` executes simultaneously the computation of the cost function
# # (or functional) and its gradient for each source based on the ``my_ensemble`` configuration. Subsequently,
# # it returns the sum of these computations, which are input to the optimisation method.
# #
# # **Steps 4-6**: We can now to obtain the predicted velocity model using the L-BFGS-B method::

# # c_optimised = minimize(J_hat, method="L-BFGS-B", options={"disp": True, "maxiter": 1},
# #                         bounds=(1.5, 3.5), derivative_options={"riesz_representation": 'l2'}
# #                         )

# # The ``minimize`` function executes the optimisation algorithm until the stopping criterion (``maxiter``) is met.
# # For 10 iterations, the predicted velocity model is shown in the following figure.
# #
# # .. image:: c_predicted.png
# #     :scale: 70 %
# #     :alt: optimised velocity model
# #     :align: center
# #
# # .. warning::
# #
# #     The ``minimize`` function employs the SciPy Python library. However, for scenarios requiring higher levels
# #     of spatial parallelism, you should evaluate how SciPy works and whether it is the best option for your problem.
# #     In addition, we apply ``derivative_options={"riesz_representation": 'l2'}`` to avoid erroneous
# #     computations in the SciPy optimisation algorithm, which is not an inner-product-aware implementation. 
# #
# # .. note::
# #
# #     This example is only a starting point to help you to tackle more intricate FWI problems.
# #
# # .. rubric:: References
# #
# # .. bibliography:: demo_references.bib
# #    :filter: docname in docnames
