import numpy as np
from devito import configuration
from examples.seismic import demo_model, plot_velocity, plot_perturbation
from examples.seismic.acoustic import AcousticWaveSolver
import time
configuration['log-level'] = 'WARNING'

nshots = 1  # Number of shots to create gradient from
nreceivers = 101  # Number of receiver locations per shot 


# Define true and initial model
shape = (101, 101)  # Number of grid point (nx, nz)
spacing = (10., 10.)  # Grid spacing in m. The domain size is now 1km by 1km
origin = (0., 0.)  # Need origin to define relative source and receiver locations

model = demo_model('circle-isotropic', vp_circle=2.5, vp_background=2.5,
                    origin=origin, shape=shape, spacing=spacing, nbl=0)

from examples.seismic import AcquisitionGeometry

t0 = 0.
tn = 1000. 
f0 = 0.010
# First, position source centrally in all dimensions, then set depth
src_coordinates = np.empty((1, 2))
src_coordinates[0, :] = np.array(model.domain_size) * .5
src_coordinates[0, 0] = 20.  # Depth is 20m


# Define acquisition geometry: receivers

# Initialize receivers for synthetic and imaging data
rec_coordinates = np.empty((nreceivers, 2))
rec_coordinates[:, 1] = np.linspace(0, model.domain_size[0], num=nreceivers)
rec_coordinates[:, 0] = 980.

# Geometry

geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, t0, tn, f0=f0, src_type='Ricker')

solver = AcousticWaveSolver(model, geometry, space_order=4)
start = time.time()
true_d, _, _ = solver.forward(vp=model.vp)
print("Time to compute true data: ", time.time()-start)
print(model.dt)