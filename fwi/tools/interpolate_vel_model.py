import firedrake as fire
from firedrake.__future__ import interpolate
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator


def model_interpolate(model, mesh, V, guess=False, name="c"):
    """Read and interpolate a seismic velocity model stored
    in a HDF5 file onto the nodes of a finite element space.

    Parameters
    ----------
    model: `dictionary`
        Model options and parameters.
    mesh: Firedrake.mesh object
        A mesh object read in by Firedrake.
    V: Firedrake.FunctionSpace object
        The space of the finite elements.
    guess: boolean, optinal
        Is it a guess model or a `exact` model?

    Returns
    -------
    c: Firedrake.Function
        P-wave seismic velocity interpolated onto the nodes of the finite elements.

    """
    sd = V.mesh().geometric_dimension()
    m = V.ufl_domain()
    if model["BCs"]["status"]:
        minz = -model["mesh"]["Lz"] - model["BCs"]["lz"]
        maxz = 0.0
        minx = 0.0 - model["BCs"]["lx"]
        maxx = model["mesh"]["Lx"] + model["BCs"]["lx"]
        miny = 0.0 - model["BCs"]["ly"]
        maxy = model["mesh"]["Ly"] + model["BCs"]["ly"]
    else:
        minz = -model["mesh"]["Lz"]
        maxz = 0.0
        minx = 0.0
        maxx = model["mesh"]["Lx"]
        miny = 0.0
        maxy = model["mesh"]["Ly"]

    W = fire.VectorFunctionSpace(m, V.ufl_element())
    coords = fire.assemble(interpolate(m.coordinates, W))
    # (z,x) or (z,x,y)
    if sd == 2:
        qp_z, qp_x = coords.dat.data[:, 0], coords.dat.data[:, 1]
    elif sd == 3:
        qp_z, qp_x, qp_y = (
            coords.dat.data[:, 0],
            coords.dat.data[:, 1],
            coords.dat.data[:, 2],
        )
    else:
        raise NotImplementedError

    if guess:
        fname = model["mesh"]["initmodel"]
    else:
        fname = model["mesh"]["truemodel"]

    with h5py.File(fname, "r") as f:
        Z = np.asarray(f.get("velocity_model")[()])

        if sd == 2:
            nrow, ncol = Z.shape
            z = np.linspace(minz, maxz, nrow)
            x = np.linspace(minx, maxx, ncol)

            # make sure no out-of-bounds
            qp_z2 = [minz if z < minz else maxz if z > maxz else z for z in qp_z]
            qp_x2 = [minx if x < minx else maxx if x > maxx else x for x in qp_x]

            interpolant = RegularGridInterpolator((z, x), Z)
            tmp = interpolant((qp_z2, qp_x2))
        elif sd == 3:
            nrow, ncol, ncol2 = Z.shape
            z = np.linspace(minz, maxz, nrow)
            x = np.linspace(minx, maxx, ncol)
            y = np.linspace(miny, maxy, ncol2)

            # make sure no out-of-bounds
            qp_z2 = [minz if z < minz else maxz if z > maxz else z for z in qp_z]
            qp_x2 = [minx if x < minx else maxx if x > maxx else x for x in qp_x]
            qp_y2 = [miny if y < miny else maxy if y > maxy else y for y in qp_y]

            interpolant = RegularGridInterpolator((z, x, y), Z)
            tmp = interpolant((qp_z2, qp_x2, qp_y2))

    c = fire.Function(V, name=name)
    c.dat.data[:] = tmp
    c = _check_units(c)
    return c


def _check_units(c):
    """Checks if velocity is in m/s or km/s"""
    if min(c.dat.data[:]) > 100.0:
        # data is in m/s but must be in km/s
        if fire.COMM_WORLD.rank == 0:
            print("INFO: converting from m/s to km/s", flush=True)
        c.assign(c / 1000.0)  # meters to kilometers
    return c