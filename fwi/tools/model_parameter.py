import numpy as np

path = '/Users/ddolci/work/firedrake/simulations_worflows_did/fwi/'
model = {}
model["mesh"] = {
    "Lz": 3.5,  # depth in km - always positive
    "Lx": 17.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": path + "inputs/marmousi.msh",
    "initmodel": path + "inputs/marmousi.hdf5",
    "truemodel": path + "inputs/marmousi.hdf5",
}
model["BCs"] = {
    "status": True,  # True or false
    "outer_bc": "non-reflective",  # None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 4.5,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": 0.9,  # thickness of the PML in the z-direction (km) - always positive
    "lx": 0.9,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
}

model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 5,
    "source_pos": np.linspace((-0.01, 14.0), (-0.01, 15.0), 10),
    "frequency": 7.0,
    "delay": 1.0,
    "num_receivers": 500,
    "receiver_locations": np.linspace((-0.10, 0.1), (-0.10, 17.0), 500),
}
model["timeaxis"] = {
    "t0": 0.0,  # Initial time for event
    "tf": 4.0,  # Final time for event
    "dt": 0.001,
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 1000,  # how frequently to output solution to pvds
    "fspool": 50,  # how frequently to save solution to RAM
}
model["path"] = path