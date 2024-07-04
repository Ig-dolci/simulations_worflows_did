from matplotlib import pyplot as plt
import numpy as np
from firedrake import *
from firedrake.pyplot import tricontourf
from tools import model
# use latex for the labels
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# increase the font size
# plt.rcParams.update({'font.size': 12})
# # read a npy file
# c_true = np.load(model["path"] + "outputs/misfit_data_2.npy")
# # plot a two-dimensional array
# # x-axis are the receivers and y-axis are the time steps
# # read a image in png format
# plt.imshow(c_true, aspect="auto", cmap="gray")
# # convert the y-axis to time steps in seconds
# plt.yticks(np.arange(0, c_true.shape[0], 1000), np.arange(0, 4.01, 1))
# # add axis labels
# plt.xlabel(r"Number of receivers")
# plt.ylabel(r"Time step (s)")
# plt.colorbar()
# plt.show()


# with CheckpointFile(model["path"] + "outputs/vel_models.h5", mode='r') as afile:
#     mesh = afile.load_mesh("mesh")
#     f = afile.load_function(mesh, "c_true")
#     g = afile.load_function(mesh, "c_guess")

# with CheckpointFile(model["path"] + "outputs/unp1.h5", mode='r') as afile:
#     u_np1 = afile.load_function(mesh, "u_np1")

# # plot the true and guess models wiht some red dots for the source locations
# plt.figure()
# # transpose the array to have the correct orientation
# f_np1 = f.dat.data_ro.T
# tricontourf(u_np1, levels=100, cmap="seismic")
# # source_locations = np.linspace(-0.01, 15.0, 10)
# # plt.scatter(source_locations, np.ones_like(source_locations), color="green", label="Source locations")
# # plt.title("True model")
# # traspose the matplotlib image
# # plt.gca().invert_yaxis()
# # transpose the labels to have the correct orientation
# plt.xlabel(r"$x$ (km)")
# plt.ylabel(r"$z$ (km)")
# plt.show()
# plt.figure()

# read a paraview txt file with the 2d wavefield
data = np.loadtxt(model["path"] + "outputs/data_plot_paraview.csv", delimiter=",")
vel_model = data[:, 0]
coordinates = data[:, 2]
# reshape the data to have the correct shape for size 299875
vel_model = vel_model.reshape(500, 599)
coordinates = coordinates.reshape(500, 599)
# plot the data
plt.figure()
plt.imshow(vel_model, aspect="auto", cmap="seismic")
plt.colorbar()
plt.show()