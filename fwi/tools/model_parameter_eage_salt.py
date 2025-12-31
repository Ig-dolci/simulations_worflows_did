import numpy as np

# EAGE Salt 3D Model parameters
# Based on SEP header: n1=676, n2=676, n3=210, d1=d2=d3=20m
# Domain dimensions: 13.52km x 13.52km x 4.2km

path = "/Users/ddolci/work/my_runs/spyro_meshes/Salt_Model_3D/3-D_Salt_Model/VEL_GRIDS/"

model = {}
model["mesh"] = {
    "xmin": 0.0,
    "xmax": 13520.0,  # 676 * 20m = 13,520m
    "ymin": 0.0,
    "ymax": 13520.0,  # 676 * 20m = 13,520m
    "zmax": 0.0,
    "zmin": -4200.0,  # 210 * 20m = 4,200m (negative for depth)
    "vp": path + "Saltf@@",  # Raw binary velocity file
    "nx": 676,  # Number of grid points in x
    "ny": 676,  # Number of grid points in y
    "nz": 210,  # Number of grid points in z
    "dx": 20.0,  # Grid spacing in x (meters)
    "dy": 20.0,  # Grid spacing in y (meters)
    "dz": 20.0,  # Grid spacing in z (meters)
}

model["BCs"] = {
    "status": True,
    "outer_bc": "non-reflective",
    "damping_type": "polynomial",
    "exponent": 2,
    "cmax": 4.6,  # Max velocity in model is 4481 m/s (~4.5 km/s)
    "R": 1e-6,  # Theoretical reflection coefficient
    "lz": 500.0,  # PML thickness in z-direction (meters)
    "lx": 500.0,  # PML thickness in x-direction (meters)
    "ly": 500.0,  # PML thickness in y-direction (meters)
}

model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 1,
    "source_pos": [(6760.0, 6760.0, -100.0)],  # Center of domain, 100m depth
    "frequency": 7.0 / 1000,  # Dominant frequency in Hz
    "delay": 1.0,
    "num_receivers": 100,
    "receiver_locations": np.array([
        [x, 6760.0, -100.0] for x in np.linspace(1000.0, 12520.0, 100)
    ]),  # Line of receivers across x at center y
}

model["timeaxis"] = {
    "t0": 0.0,
    "tf": 4000.0,  # Final time in seconds
    "dt": 0.1,  # Time step in seconds (1 ms)
    "amplitude": 1,
    "nspool": 500,
    "fspool": 500,
}

model["path"] = path
