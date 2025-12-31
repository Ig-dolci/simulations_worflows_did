from tools.wave_parameter import parameter_interpolate, read_segy
from tools.model_parameter import model
import numpy as np
from firedrake import *

M = 1
my_ensemble = Ensemble(COMM_WORLD, M)
num_sources = my_ensemble.ensemble_comm.size
source_number = my_ensemble.ensemble_comm.rank

file = "/Users/ddolci/work/my_runs/simulations_worflows_did/fwi/inputs/MODEL_P-WAVE_VELOCITY_1.25m.segy"
vp = read_segy(file)
vp = vp.T
mesh = Mesh(
    model["mesh"]["meshfile"], comm=my_ensemble.comm,
    distribution_parameters={
                "overlap_type": (DistributedMeshOverlapType.NONE, 0)
            }, name="mesh"
)

element = FiniteElement("KMV", mesh.ufl_cell(), degree=6)
V0 = FunctionSpace(mesh, element)
vp_true = parameter_interpolate(model, V0, vp, l_grid=1.25/1000, name="vp_true")
VTKFile("outputs/vp_true0.pvd").write(vp_true)
