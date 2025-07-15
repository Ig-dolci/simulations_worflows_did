import numpy as np

path = "/Users/ddolci/work/my_runs/simulations_worflows_did/fwi/inputs/"
path1 = '/Users/ddolci/work/my_runs/generate_marmousi_mesh/'
model = {}
model["mesh"] = {
    "xmin": 6.0,
    "xmax": 13.0,
    "zmax": 0.0,
    "zmin": -3.5,
    "meshfile": path1 + "mm_acoustic.msh",
    "vs": path + "MODEL_S-WAVE_VELOCITY_1.25m.segy",
    "vp": path + "MODEL_P-WAVE_VELOCITY_1.25m.segy",
    "density": path + "MODEL_DENSITY_1.25m.segy",
}
model["BCs"] = {
    "status": True,  # True or false
    "outer_bc": "non-reflective",  # None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 4.5,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": 0.5,  # thickness of the PML in the z-direction (km) - always positive
    "lx": 0.5,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 0.5,  # thickness of the PML in the y-direction (km) - always positive
}

model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 4,
    "source_pos": np.linspace((7, -0.1), (13, -0.1), 4),
    "frequency": 10.,
    "delay": 1.0,
    "num_receivers": 10,
    "receiver_locations": np.linspace((7, -0.15), (13, -0.15), 10),
}

model["timeaxis"] = {
    "t0": 0.0,  # Initial time for event
    "tf": 1.,  # Final time for event
    "dt": 0.001,  # time step
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 1000,  # how frequently to output solution to pvds
    "fspool": 500,  # how frequently to save solution to RAM
}

model["path"] = path
