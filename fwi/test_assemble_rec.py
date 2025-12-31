from firedrake import *
from firedrake.__future__ import interpolate
import numpy as np
import time

mesh = UnitSquareMesh(50, 50)
receiver_locations = np.linspace((0.2, 0.8), (0.8, 0.8), 20)

V = FunctionSpace(mesh, "KMV", 1)
u_np1 = Function(V).assign(1.0)

receiver_mesh = VertexOnlyMesh(mesh, receiver_locations)
V_r = FunctionSpace(receiver_mesh, "DG", 0)

interpolate_receivers = interpolate(u_np1, V_r)

start = time.time()
for i in range(100):
    assemble(interpolate_receivers)
end = time.time()
print("Time to assemble: ", end-start)