import numpy as np
import json
import os
import matplotlib.pylab as plt
from matplotlib.widgets import Slider

np.random.seed(1)


def sin_3d_fct(x, y, z):
    data = np.sin(0.0500 * np.linalg.norm(np.array([x, y, z]), axis=0))
    return data



def plot_3D(xy_coords, depth, coords_field, function):
    """
    Plots the results for a 2D field

    :param output_name: file name of the plot
    :param idx: index of the 3D plot in a 2D field
    :param validation: (optional) validation data. if not False the validation data and error are plotted.
                       Default is None. The size of the validation dataset must be the same as the interpolated
    :param show: bool to show the figure (default True). if set to false saves the figure
    :return:
    """

    z_coord = depth
    data_points_z = function(xy_coords[:, 0], xy_coords[:, 1], z_coord[0])

    data_points_field = function(coords_field[0], coords_field[1], coords_field[2])

    fig, ax = plt.subplots(2, 1, figsize=(6.5, 9), sharex=True, sharey=True)
    ax[0].set_position([0.15, 0.58, 0.5, 0.35])
    ax[1].set_position([0.15, 0.18, 0.5, 0.35])

    ax[0].set_title(f"Depth {z_coord[0]}")
    im1 = ax[0].scatter(xy_coords[:, 0], xy_coords[:, 1], c=data_points_z, vmin=np.min(data_points_z), vmax=np.max(data_points_z), cmap="jet",
                        edgecolor='k', marker="o")
    im2 = ax[1].pcolor(coords_field[0][:, :, 0], coords_field[1][:, :, 0], data_points_field[:, :, 0],
                 vmin=np.min(data_points_z), vmax=np.max(data_points_z), cmap="jet")
    im3 = ax[1].scatter(xy_coords[:, 0], xy_coords[:, 1], c=data_points_z, vmin=np.min(data_points_z), vmax=np.max(data_points_z), cmap="jet",
                        edgecolor='k', marker="o")

    cbar = plt.colorbar(im1, ax=ax[0])
    cbar.ax.set_ylabel('Data')
    ax[0].grid()
    ax[0].set_ylabel("Y coordinate")

    cbar = plt.colorbar(im2, ax=ax[1])
    cbar.ax.set_ylabel('Data')
    ax[1].grid()
    ax[1].set_ylabel("Y coordinate")
    ax[1].set_xlabel("X coordinate")

    ax_slider = plt.axes([0.92, 0.35, 0.025, 0.25])
    slider = Slider(
        ax=ax_slider,
        label='Depth',
        valmin=np.amin(z_coord),
        valmax=np.amax(z_coord),
        valinit=z_coord[0],
        orientation="vertical"
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        ax[0].set_title(f"Depth {round(slider.val, 2)}")
        ax[0].scatter(xy_coords[:, 0], xy_coords[:, 1],
                      c=function(xy_coords[:, 0], xy_coords[:, 1], slider.val),
                      vmin=np.min(data_points_z), vmax=np.max(data_points_z), cmap="jet",
                      edgecolor='k', marker="o")

        idx = np.argmin(np.abs(coords_field[2][0, :, :][0] - slider.val))
        ax[1].pcolor(coords_field[0][:, :, idx], coords_field[1][:, :, idx], data_points_field[:, :, idx],
                     vmin=np.min(data_points_z), vmax=np.max(data_points_z), cmap="jet")

        ax[1].scatter(xy_coords[:, 0], xy_coords[:, 1],
                      c=function(xy_coords[:, 0], xy_coords[:, 1], slider.val),
                      vmin=np.min(data_points_z), vmax=np.max(data_points_z), cmap="jet",
                      edgecolor='k', marker="o")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

    return


def main(dimensions=[10, 10, 10], discretisation=[11, 11, 11], nb_samples=5, output="./"):
    """
    Create dataset based on analytical solution

    :param x: x coordinate (max and min)
    :param y: y coordinate (max and min)
    :param z: z coordinate (max and min)
    :param output: location of the output files
    :param nb_points: list number of points for interpolation on the xy plane
    :return:
    """

    # if path does not exist -> creates
    if not os.path.isdir(output):
        os.makedirs(output)

    # create meshgrid arrays
    gx, gy, gz = np.meshgrid(np.linspace(-dimensions[0], dimensions[0], discretisation[0]),
                             np.linspace(-dimensions[1], dimensions[1], discretisation[1]),
                             np.linspace(-dimensions[2], dimensions[2], discretisation[2]),
                             indexing='ij')
    # data 3D
    data_curve = sin_3d_fct(gx.ravel(), gy.ravel(), gz.ravel())
    coord_3d = np.array([gx.ravel(), gy.ravel(), gz.ravel()]).T

    # 2D plane
    xy_plane = np.unique(np.array([gx.ravel(), gy.ravel()]).T, axis=0)
    depth = np.linspace(0, dimensions[2], discretisation[2])

    # select random points from data_curve according to nb_points
    idx_xy_plane = np.random.choice(len(xy_plane), size=nb_samples, replace=False)

    training_points = []
    training_data = []
    for idx in idx_xy_plane:
        tr_id = np.where((coord_3d[:, 0] == xy_plane[idx, 0]) & (coord_3d[:, 1] == xy_plane[idx, 1]))[0]
        training_points.append(xy_plane[idx, :])
        training_data.append(data_curve[tr_id])

    # results
    results = {"validation": {"x-coord":  gx.ravel().tolist(),
                              "y-coord": gy.ravel().tolist(),
                              "z-coord": gz.ravel().tolist(),
                              "data": [i for i in data_curve]},
               "training": {"x-coord": np.array(training_points)[:, 0].tolist(),
                            "y-coord": np.array(training_points)[:, 1].tolist(),
                            "z-coord": depth.tolist(),
                            "data": [t.tolist() for t in training_data]
                            }}

    # dump file to pickle
    with open(os.path.join(output, "sin_data_3d.json"), "w") as fo:
        json.dump(results, fo, indent=2)

    # # create plot
    plot_3D(np.array(training_points), depth, [gx, gy, gz], sin_3d_fct)

    return


if __name__ == "__main__":
    main(dimensions=[20, 20, 10], discretisation=[21, 21, 41], nb_samples=10, output="../data/processed")
