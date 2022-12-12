from gstools import SRF, Gaussian
import numpy as np
import matplotlib.pyplot as plt
import csv
from itertools import zip_longest
import random
from scipy.spatial import ConvexHull, Delaunay
import copy
import plotly.graph_objects as go
import pandas


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


def generate_random_surfaces(n_layers):
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
            sorted(random.sample(range(dim), n_layers - 1))
            for i in range(points_for_surface)
        ]
    )
    z_points = z_points.T
    surfaces[1:-1] = [
        surface_points_from_depth(x_coord, y_coord, z) for z in z_points
    ]
    return surfaces


if __name__ == "__main__":
    # coords
    dim = 64
    dim_y = 500
    x_coord = np.linspace(1, dim, dim, dtype=int)
    y_coord = np.linspace(1, dim_y, dim_y, dtype=int)
    z_coord = np.linspace(1, dim, dim, dtype=int)
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
    
    surfaces = generate_random_surfaces(4)

    for counter_s, seed in enumerate(range(101010, 101110, 1)):
        print(counter_s)
        random.seed(seed)
        srf_sand = create_random_fields(
            0.3, 1.5, v_min, v_max, aniso_x, aniso_z, ndim, lognormal, seed + 1
        )
        srf_clay = create_random_fields(
            0.3, 2, v_min, v_max, aniso_x, aniso_z, ndim, lognormal, seed + 2
        )
        srf_silt = create_random_fields(
            0.5, 3, v_min, v_max, aniso_x, aniso_z, ndim, lognormal, seed + 3
        )
        layers = [srf_clay, srf_silt,srf_clay, srf_sand]


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

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=xs.flatten(), y=ys.flatten(), z=zs.flatten(), marker=dict(color=values.flatten()), mode='markers'))
        fig.write_html(f"data\\layers_n\\test_{counter_s}.html")

        d = [xs.flatten(), ys.flatten(), zs.flatten(), values.flatten()]

        #export_data = zip_longest(*d, fillvalue="")
        #with open(
        #    f"data\\layers_n\\RF_3d_{counter_s}.csv", "w", encoding="ISO-8859-1", newline=""
        #) as myfile:
        #    wr = csv.writer(myfile)
        #    wr.writerow(("x", "y", "z", "IC"))
        #    wr.writerows(export_data)
        #myfile.close()

        df = pandas.DataFrame({"x": xs.flatten(), 
                               "y": ys.flatten(), 
                               "z": zs.flatten(), 
                               "IC": values.flatten()})
        
        grouped = df.groupby('y')
        test_group = np.random.choice(range(len(grouped)),  int(len(grouped) * 0.2), replace=False)
        for name, group in grouped: 
            plt.clf()
            plt.imshow(np.reshape(list(group['IC']), (64, 64)))
            if name in test_group:
                plt.savefig(f"data\\layers_n\\test\\2d\\RF_2d_{counter_s}_{name}.png")
                group.to_csv(f"data\\layers_n\\test\\2d\\RF_2d_{counter_s}_{name}.csv")                
            else:
                plt.savefig(f"data\\layers_n\\train\\2d\\RF_2d_{counter_s}_{name}.png")
                group.to_csv(f"data\\layers_n\\train\\2d\\RF_2d_{counter_s}_{name}.csv")