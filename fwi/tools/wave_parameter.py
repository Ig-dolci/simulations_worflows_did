import firedrake as fire
from firedrake.__future__ import interpolate
import numpy as np
import h5py
import os
import segyio
from scipy.interpolate import RegularGridInterpolator
import scipy.ndimage
import matplotlib.pyplot as plt


def parameter_interpolate(
        model, function_space, parameter, l_grid, name="c", smoth_par=False,
        **kwargs
):
    """Read and interpolate a seismic parameter model stored
    in a HDF5 file onto the nodes of a finite element space.

    Parameters
    ----------
    model: `dictionary`
        Model options and parameters.
    function_space: `firedrake.FunctionSpace`
        Function space where the model will be interpolated.
    parameter: numpy.ndarray
        Parameter to be interpolated.
    l_grid: `float`
        Grid spacing of the model.
    name: `str`, optional
        Name of the parameter.
    smoth_par: `bool`, optional
        Apply a Gaussian filter to the model parameter.
    **kwargs: `dict`
        Additional keyword arguments for the Gaussian filter.

    Returns
    -------
    c: Firedrake.Function
        Wave equations parameter interpolated onto the nodes of the finite elements.
        For example, the seismic parameter model for the acoustic wave equation.

    """
    sd = function_space.mesh().geometric_dimension()
    m = function_space.ufl_domain()
    zmin = model["mesh"]["zmin"]
    zmax = model["mesh"]["zmax"]
    xmin = model["mesh"]["xmin"]
    xmax = model["mesh"]["xmax"]
    if sd == 3:
        ymin = model["mesh"]["ymin"]
        ymax = model["mesh"]["ymax"]
    if model["BCs"]["status"]:
        if zmin < 0.0:
            # The depth is in z-direction
            zmin -= model["BCs"]["lz"]
        if xmin < 0.0:
            # The depth is in x-direction
            xmin -= model["BCs"]["lx"]
        if zmax > 0.0 and zmin >= 0.0:
            zmin -= model["BCs"]["lz"]
            zmax += model["BCs"]["lz"]
        if xmax > 0.0 and xmin >= 0.0:
            xmin -= model["BCs"]["lx"]
            xmax += model["BCs"]["lx"]
        if sd == 3:
            ymin -= model["BCs"]["ly"]
            ymax += model["BCs"]["ly"]

    W = fire.VectorFunctionSpace(m, function_space.ufl_element())
    coords = fire.assemble(interpolate(m.coordinates, W))

    if sd == 2:
        qp_x, qp_z = coords.dat.data[:, 0], coords.dat.data[:, 1]
    elif sd == 3:
        qp_z, qp_x, qp_y = (
            coords.dat.data[:, 0],
            coords.dat.data[:, 1],
            coords.dat.data[:, 2],
        )
    else:
        raise NotImplementedError("Only 2D and 3D models are supported")

    # This enable the user cut the model to the desired region.
    if zmin < 0.0:
        Z0 = parameter[
            int(model["mesh"]["xmin"]/l_grid):int(model["mesh"]["xmax"]/l_grid),
            -int(model["mesh"]["zmax"]/l_grid):-int(model["mesh"]["zmin"]/l_grid)
        ]
    if xmin < 0.0:
        Z0 = parameter[
            -int(model["mesh"]["xmax"]/l_grid):-int(model["mesh"]["xmin"]/l_grid),
            int(model["mesh"]["zmin"]/l_grid):int(model["mesh"]["zmax"]/l_grid)
        ]
    Z = np.zeros((int((xmax - xmin)/l_grid), int((zmax - zmin)/l_grid)))
    if model["BCs"]["status"]:
        if zmin < 0.0:
            # The depth is in z-direction
            Z[
                int(model["BCs"]["lx"]/l_grid):-int(model["BCs"]["lx"]/l_grid),
                :-int(model["BCs"]["lz"]/l_grid)
            ] = Z0
            for i in range(int(model["BCs"]["lx"]/l_grid) + 1):
                Z[-i, 0:-int(model["BCs"]["lz"]/l_grid)] = Z0[-1, :]
                Z[i, 0:-int(model["BCs"]["lz"]/l_grid)] = Z0[0, :]
            for i in range(int(model["BCs"]["lz"]/l_grid) + 1):
                Z[:, -i] = Z[:, -int(model["BCs"]["lz"]/l_grid)-1]

        if xmin < 0.0:
            # The depth is in x-direction
            Z[
                :-int(model["BCs"]["lx"]/l_grid),
                int(model["BCs"]["lz"]/l_grid):-int(model["BCs"]["lz"]/l_grid)
            ] = Z0
            for i in range(int(model["BCs"]["lz"]/l_grid) + 1):
                Z[0:-int(model["BCs"]["lx"]/l_grid), -i] = Z0[:, -1]
                Z[0:-int(model["BCs"]["lx"]/l_grid), i] = Z0[:, 0]
            for i in range(int(model["BCs"]["lx"]/l_grid) + 1):
                Z[-i, int(model["BCs"]["lz"]/l_grid):-int(model["BCs"]["lz"]/l_grid)] = Z0[-1, :]

        if smoth_par:
            Z = scipy.ndimage.gaussian_filter(Z, sigma=kwargs.get("sigma", 50))
        if sd == 2:
            nrow, ncol = Z.shape
            if zmin < 0.0:
                z = np.linspace(zmax, zmin, ncol)
            else:
                z = np.linspace(zmin, zmax, ncol)
            if xmin < 0.0:
                x = np.linspace(xmax, xmin, nrow)
            else:
                x = np.linspace(xmin, xmax, nrow)

            # make sure no out-of-bounds
            qp_z2 = [zmin if z < zmin else zmax if z > zmax else z for z in qp_z]
            qp_x2 = [xmin if x < xmin else xmax if x > xmax else x for x in qp_x]
            interpolant = RegularGridInterpolator((x, z), Z)
            tmp = interpolant((qp_x2, qp_z2))
        elif sd == 3:
            nrow, ncol, ncol2 = Z.shape
            z = np.linspace(zmin, zmax, nrow)
            x = np.linspace(xmin, xmax, ncol)
            y = np.linspace(ymin, ymax, ncol2)

            # make sure no out-of-bounds
            qp_z2 = [zmin if z < zmin else zmax if z > zmax else z for z in qp_z]
            qp_x2 = [xmin if x < xmin else xmax if x > xmax else x for x in qp_x]
            qp_y2 = [ymin if y < ymin else ymax if y > ymax else y for y in qp_y]

            interpolant = RegularGridInterpolator((z, x, y), Z)
            tmp = interpolant((qp_z2, qp_x2, qp_y2))

    c = fire.Function(function_space, name=name)
    c.dat.data[:] = tmp
    c = _check_units(c)
    return c


def _check_units(c):
    """Checks if parameter is in m/s or km/s"""
    if max(c.dat.data[:]) > 1000.0:
        # data is in m/s but must be in km/s
        if fire.COMM_WORLD.rank == 0:
            print("INFO: converting from m/s to km/s", flush=True)
        c.assign(c / 1000.0)  # meters to kilometers
    return c


def read_segy(input_filename, save_figure=False):
    """Reads a segy file and returns the data as a numpy array.

    Parameters
    ----------
    input_filename : string
        The name of the input file.

    Returns
    -------
    vp : numpy.ndarray
        The data read from the segy file.

    """
    f, filetype = os.path.splitext(input_filename)

    if filetype == ".segy":
        with segyio.open(input_filename, ignore_geometry=True) as f:
            nx, nz = len(f.samples), len(f.trace)
            vp = np.zeros(shape=(nx, nz))
            for index, trace in enumerate(f.trace):
                vp[:, index] = trace

    if save_figure:
        plt.imshow(vp, cmap="jet", aspect="auto")
        plt.colorbar()
        plt.savefig(f + ".png")
        plt.close()
    return vp
