from firedrake import *
import math


def damp_functions(model, V, mesh, dim=2):
    """Damping functions for the perfect matched layer for 2D and 3D

    Parameters
    ----------
    model : dict
        A dictionary with the parameters of the problem.

    Returns
    -------
    sigma_x : obj
        Firedrake function with the damping function in the x direction
    sigma_z : obj
        Firedrake function with the damping function in the z direction
    sigma_y : obj
        Firedrake function with the damping function in the y direction

    """

    ps = model["BCs"]["exponent"]
    cmax = model["BCs"]["cmax"]  # maximum acoustic wave velocity
    R = model["BCs"]["R"]  # theoretical reclection coefficient
    pad_length = model["BCs"]["lz"]  # thickness of the PML in the z-direction
    x, z = SpatialCoordinate(mesh)
    if model["mesh"]["zmin"] < 0.0:
        # The depth is in z-direction
        z2 = model["mesh"]["zmin"]
        z1 = model["mesh"]["zmax"]
    else:
        z1 = model["mesh"]["zmin"]
        z2 = model["mesh"]["zmax"]
    if model["mesh"]["xmin"] < 0.0:
        # The depth is in x-direction
        x2 = model["mesh"]["xmin"]
        x1 = model["mesh"]["xmax"]
    else:
        x1 = model["mesh"]["xmin"]
        x2 = model["mesh"]["xmax"]

    bar_sigma = ((3.0 * cmax) / (2.0 * pad_length)) * math.log10(1.0 / R)
    aux1 = Function(V)
    aux2 = Function(V)
    if z2 < 0.0:
        # Sigma X
        sigma_max_x = bar_sigma  # Max damping
        aux1.interpolate(
            conditional(
                And((x >= x1 - pad_length), x < x1),
                ((abs(x - x1) ** (ps)) / (pad_length ** (ps))) * sigma_max_x,
                0.0 
            )
        )
        aux2.interpolate(
            conditional(
                And(x > x2, (x <= x2 + pad_length)),
                ((abs(x - x2) ** (ps)) / (pad_length ** (ps))) * sigma_max_x,
                0.0,
            )
        )
        sigma_x = Function(V, name="sigma_x").interpolate(aux1 + aux2)

        # Sigma Z
        tol_z = 1.000001
        sigma_max_z = bar_sigma  # Max damping
        aux1.interpolate(
            conditional(
                And(z >= z2 - tol_z * pad_length, (z < z2)),
                ((abs(z - z2) ** (ps)) / (pad_length ** (ps))) * sigma_max_z,
                0.0,
            )
        )

        sigma_z = Function(V, name="sigma_z").interpolate(aux1)
    if x2 < 0.0:
        raise NotImplementedError("Not implemented for depth in x-direction")
    # VTKFile("outputs/sigma_z.pvd").write(sigma_z)
    # VTKFile("outputs/sigma_x.pvd").write(sigma_x)
    # quit()
    if dim == 2:
        return (sigma_x, sigma_z)

    elif dim == 3:
        # Sigma Y
        sigma_max_y = bar_sigma  # Max damping
        y = Wave_obj.mesh_y
        y1 = 0.0
        y2 = Wave_obj.length_y
        aux1.interpolate(
            conditional(
                And((y >= y1 - pad_length), y < y1),
                ((abs(y - y1) ** (ps)) / (pad_length ** (ps))) * sigma_max_y,
                0.0,
            )
        )
        aux2.interpolate(
            conditional(
                And(y > y2, (y <= y2 + pad_length)),
                ((abs(y - y2) ** (ps)) / (pad_length ** (ps))) * sigma_max_y,
                0.0,
            )
        )
        sigma_y = Function(V, name="sigma_y").interpolate(aux1 + aux2)
        # sgm_y = File("pmlField/sigma_y.pvd")
        # sgm_y.write(sigma_y)

        return (sigma_x, sigma_y, sigma_z)

