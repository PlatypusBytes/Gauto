import numpy as np
import json
import os
import matplotlib.pylab as plt
from matplotlib.widgets import Slider

np.random.seed(1)


def sin_3d_fct(x, y, z):
    data = np.sin(0.0500 * np.linalg.norm(np.array([x, y, z]), axis=0))
    return data



def plot_3D(coords, coords_field, function):
    """
    Plots the results for a 2D field

    :param output_name: file name of the plot
    :param idx: index of the 3D plot in a 2D field
    :param validation: (optional) validation data. if not False the validation data and error are plotted.
                       Default is None. The size of the validation dataset must be the same as the interpolated
    :param show: bool to show the figure (default True). if set to false saves the figure
    :return:
    """

    z_coord = np.sort(np.array(list(set(coords[:, 2]))))
    data_points_z = function(coords[:, 0], coords[:, 1], z_coord[0])

    data_points_field = function(coords_field[0], coords_field[1], coords_field[2])

    fig, ax = plt.subplots(2, 1, figsize=(6.5, 9), sharex=True, sharey=True)
    ax[0].set_position([0.15, 0.58, 0.5, 0.35])
    ax[1].set_position([0.15, 0.18, 0.5, 0.35])

    ax[0].set_title(f"Depth {z_coord[0]}")
    im1 = ax[0].scatter(coords[:, 0], coords[:, 1], c=data_points_z, vmin=np.min(data_points_z), vmax=np.max(data_points_z), cmap="jet",
                        edgecolor='k', marker="o")
    im2 = ax[1].pcolor(coords_field[0][:, :, 0], coords_field[1][:, :, 0], data_points_field[:, :, 0],
                 vmin=np.min(data_points_z), vmax=np.max(data_points_z), cmap="jet")
    im3 = ax[1].scatter(coords[:, 0], coords[:, 1], c=data_points_z, vmin=np.min(data_points_z), vmax=np.max(data_points_z), cmap="jet",
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
        ax[0].scatter(coords[:, 0], coords[:, 1],
                      c=function(coords[:, 0], coords[:, 1], slider.val),
                      vmin=np.min(data_points_z), vmax=np.max(data_points_z), cmap="jet",
                      edgecolor='k', marker="o")

        idx = np.argmin(np.abs(coords_field[2][0, :, :][0] - slider.val))
        ax[1].pcolor(coords_field[0][:, :, idx], coords_field[1][:, :, idx], data_points_field[:, :, idx],
                     vmin=np.min(data_points_z), vmax=np.max(data_points_z), cmap="jet")

        ax[1].scatter(coords[:, 0], coords[:, 1],
                      c=function(coords[:, 0], coords[:, 1], slider.val),
                      vmin=np.min(data_points_z), vmax=np.max(data_points_z), cmap="jet",
                      edgecolor='k', marker="o")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

    return


def main(x=10, y=10, z=10, output="./", nb_points=[100, 100, 50]):
    """
    Create dataset based on analytical solution

    :param x: x coordinate (max and min)
    :param y: y coordinate (max and min)
    :param z: z coordinate (max and min)
    :param output: location of the output files
    :param nb_points: list number of points for interpolation along each axis
    :return:
    """

    # if path does not exist -> creates
    if not os.path.isdir(output):
        os.makedirs(output)

    # create meshgrid arrays
    gx, gy, gz = np.meshgrid(np.linspace(-x, x, nb_points[0]),
                             np.linspace(-y, y, nb_points[1]),
                             np.linspace(-z, z, nb_points[2]),
                             indexing='ij')

    data_curve = sin_3d_fct(gx.ravel(), gy.ravel(), gz.ravel())

    # select random points from data_curve according to nb_points
    id = np.random.choice(len(gx.ravel()), size=sum(nb_points), replace=False)
    coords = np.array([gx.ravel(), gy.ravel(), gz.ravel()]).T[id, :]
    data_points = sin_3d_fct(coords[:, 0], coords[:, 1], coords[:, 2])

    # results
    results = {"validation": {"x-coord": [i for i in gx.ravel()],
                              "y-coord": [i for i in gy.ravel()],
                              "z-coord": [i for i in gz.ravel()],
                              "data": [i for i in data_curve]},
               "training": {"x-coord": [i for i in coords[:, 0]],
                            "y-coord": [i for i in coords[:, 1]],
                            "z-coord": [i for i in coords[:, 2]],
                            "data": [i for i in data_points]}}

    # dump file to pickle
    with open(os.path.join(output, "sin_data_3d.json"), "w") as fo:
        json.dump(results, fo, indent=2)

    # # create plot
    plot_3D(coords, [gx, gy, gz], sin_3d_fct)

    return


if __name__ == "__main__":
    main(x=100, y=100, z=100, output="../data/processed", nb_points=[100, 75, 50])
