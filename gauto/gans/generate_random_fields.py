from gstools import SRF, Gaussian
import numpy as np
import matplotlib.pyplot as plt
import csv
from itertools import zip_longest
import random
from scipy.spatial import ConvexHull, Delaunay
import copy
import plotly.graph_objects as go


def create_random_fields(
    std_value, mean, v_min, v_max, aniso_x, aniso_z, ndim, lognormal, seed
):
    len_scale = np.array([aniso_x, aniso_x, aniso_z])
    if lognormal:
        var = np.log((std_value / mean) ** 2 + 1)
        mean = np.log(mean**2 / (np.sqrt(mean**2 + std_value**2)))
    else:
        var = std_value**2

    model = Gaussian(dim=ndim, var=var, len_scale=len_scale)
    srf = SRF(model, mean=mean, seed=seed)

    return srf


def surface_points_from_depth(x_coord, y_coord, z):
    return [
        (min(x_coord), min(y_coord), z[0]),
        (min(x_coord), max(y_coord), z[1]),
        (max(x_coord), min(y_coord), z[2]),
        (max(x_coord), max(y_coord), z[3]),
    ]


if __name__ == "__main__":
    # coords

    x_coord = np.linspace(1, 100, 100, dtype=int)
    y_coord = np.linspace(1, 256, 256, dtype=int)
    z_coord = np.linspace(1, 256, 256, dtype=int)
    xs, ys, zs = np.meshgrid(x_coord, y_coord, z_coord, indexing="ij")
    # number of RFs
    rf_number = 2

    std_value = 0.4
    mean = 2
    v_min = 1
    v_max = 3
    theta = 50
    aniso_x = 40
    aniso_z = 10
    ndim = 3
    lognormal = False
    seed = 101010

    srf_sand = create_random_fields(
        0.03, 1.5, v_min, v_max, aniso_x, aniso_z, ndim, lognormal, seed + 1
    )
    srf_clay = create_random_fields(
        0.03, 2, v_min, v_max, aniso_x, aniso_z, ndim, lognormal, seed + 2
    )
    srf_silt = create_random_fields(
        0.03, 3, v_min, v_max, aniso_x, aniso_z, ndim, lognormal, seed + 3
    )
    layers = [srf_silt, srf_clay, srf_sand]
    n_layers = 3
    points_for_surface = 4
    surfaces = np.zeros((n_layers + 1, points_for_surface, 3))
    top_surface = [
        (min(x_coord), min(y_coord), min(z_coord)),
        (min(x_coord), max(y_coord), min(z_coord)),
        (max(x_coord), min(y_coord), min(z_coord)),
        (max(x_coord), max(y_coord), min(z_coord)),
    ]
    bottom_surface = [
        (min(x_coord), min(y_coord), max(z_coord)),
        (min(x_coord), max(y_coord), max(z_coord)),
        (max(x_coord), min(y_coord), max(z_coord)),
        (max(x_coord), max(y_coord), max(z_coord)),
    ]
    surfaces[0] = top_surface
    surfaces[-1] = bottom_surface
    # create random depth values depending on the layers
    z_points = np.array(
        [
            sorted(random.sample(range(256), n_layers - 1))
            for i in range(points_for_surface)
        ]
    )
    z_points = z_points.T
    surfaces[1:-1] = [surface_points_from_depth(x_coord, y_coord, z) for z in z_points]

    all_available_points = np.array([xs.flatten(), ys.flatten(), zs.flatten()]).T
    values = np.zeros(all_available_points.shape[0])
    layer_point_locations = []
    for counter, surfaces in enumerate(zip(surfaces, surfaces[1:])):
        top_surf, bottom_surf = surfaces
        # collect points that are in this layer
        poly_points = list(top_surf) + list(bottom_surf)
        points = []

        mask = Delaunay(poly_points).find_simplex(all_available_points) >= 0
        layer_point_locations = all_available_points[mask]
        layer_IC = layers[counter](layer_point_locations.T)
        values[mask] = layer_IC

    d = [xs.flatten(), ys.flatten(), zs.flatten(), values.flatten()]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=d[0],
                y=d[1],
                z=d[2],
                mode="markers",
                marker=dict(
                    size=12,
                    color=d[3],  # set color to an array/list of desired values
                    colorscale="Viridis",  # choose a colorscale
                    opacity=0.8,
                ),
            )
        ]
    )
    fig.write_html("data\\RF_1.html")

    export_data = zip_longest(*d, fillvalue="")
    with open(f"data\\RF_1.csv", "w", encoding="ISO-8859-1", newline="") as myfile:
        wr = csv.writer(myfile)
        wr.writerow(("x", "y", "z", "IC"))
        wr.writerows(export_data)
    myfile.close()
