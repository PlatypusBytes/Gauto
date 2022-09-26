import numpy as np
import pandas as pd
import plotly.express as px
from gstools import SRF, Exponential, Gaussian, Matern, Linear
import matplotlib.pyplot as plt


def create_surface(layer, x, y, z):
    return layer["a"] * x + layer["b"] * y + layer["c"] * z + layer["d"]


def get_srf(item, value_name):
    model_name = "Gaussian"
    ndim = 3
    # set scale of fluctuation
    theta = 5
    len_scale = np.array([item["sof_theta_h"], 1, item["sof_theta_v"]]) * theta
    var = item["cov_" + value_name]
    angles = 0
    seed = 822022

    # initialise model
    if model_name == "Gaussian":
        model = Gaussian(dim=ndim, var=var, len_scale=len_scale, angles=angles)
    elif model_name == "Exponential":
        model = Exponential(dim=ndim, var=var, len_scale=len_scale, angles=angles)
    elif model_name == "Matern":
        model = Matern(dim=ndim, var=var, len_scale=len_scale, angles=angles)
    elif model_name == "Linear":
        model = Linear(dim=ndim, var=var, len_scale=len_scale, angles=angles)
    else:
        print('model name: "', model_name, '" is not supported')

    # initialise random field
    return SRF(model, mean=item["mean_" + value_name], seed=seed)


# case G1

# boundary surface
boundary_surfaces = {"G1" : {"layer_1": {"a": 0, "b": 1, "c": 0, "d": -2},
"layer_2": {"a": 0, "b": 1, "c": 0, "d": -6}},
"G2" : {"layer_1": {"a": 0.05, "b": 1, "c": -0.05, "d": -2},
"layer_2": {"a": -0.05, "b": 1, "c": 0.05, "d": -6}},
"G3" : {"layer_1": {"a": 0.05, "b": 1, "c": -0.05, "d": -2},
"layer_2": {"a": -0.05, "b": 1, "c": 0.05, "d": -6}},
"G4" : {"layer_1": {"a": 0, "b": 1, "c": 0, "d": -2},
"layer_2": {"a": 0.25, "b": 1, "c": 0.25, "d": -6}}}

for name, boundary_layers in boundary_surfaces.items():
    layer_1 = boundary_layers["layer_1"]
    layer_2 = boundary_layers["layer_2"]
    layer_points_1 = []
    layer_points_2 = []
    soils = {
        1: {
            "mean_qt": 9,
            "mean_fs": 110,
            "cov_qt": 0.24,
            "cov_fs": 0.24,
            "sof_theta_h": 10,
            "sof_theta_v": 1,
        },
        2: {
            "mean_qt": 2,
            "mean_fs": 75,
            "cov_qt": 0.2,
            "cov_fs": 0.2,
            "sof_theta_h": 15,
            "sof_theta_v": 1.2,
        },
        3: {
            "mean_qt": 5.5,
            "mean_fs": 85,
            "cov_qt": 0.24,
            "cov_fs": 0.24,
            "sof_theta_h": 10,
            "sof_theta_v": 1,
        },
    }
    
    
    srf_qt_1 = get_srf(soils[1], "qt")
    srf_qt_2 = get_srf(soils[2], "qt")
    srf_qt_3 = get_srf(soils[3], "qt")
    srf_fs_1 = get_srf(soils[1], "fs")
    srf_fs_2 = get_srf(soils[2], "fs")
    srf_fs_3 = get_srf(soils[3], "fs")
    
    points = []
    for i in range(256):
        for j in range(256):
            for k in range(100):
                point = {"x": i, "y": j, "z": k}
                print(point)
                is_above_layer_1 = (create_surface(layer_1, i, j, k) <= 0) and (
                    create_surface(layer_2, i, j, k) <= 0
                )
                is_above_layer_2 = (create_surface(layer_1, i, j, k) > 0) and (
                    create_surface(layer_2, i, j, k) < 0
                )
                is_above_layer_3 = (create_surface(layer_1, i, j, k) > 0) and (
                    create_surface(layer_2, i, j, k) < 0
                )
                if is_above_layer_1:
                    point["layer"] = 1
                    point["qt"] = srf_qt_1((i,j,k))[0]
                    point["fs"] = srf_fs_1((i,j,k))[0]
                elif is_above_layer_2:
                    point["layer"] = 2 
                    point["qt"] = srf_qt_2((i,j,k))[0]
                    point["fs"] = srf_fs_2((i,j,k))[0]   
                else:
                    point["layer"] = 3
                    point["qt"] = srf_qt_3((i,j,k))[0]
                    point["fs"] = srf_fs_3((i,j,k))[0]
    
                points.append(point)
    df = pd.DataFrame(points)
    df.to_csv("data/synthetic_data_" + name + ".csv")
    fig = px.scatter_3d(df, x="x", y="y", z="z", color="fs")
    fig.write_html("data/" + name + ".html")
    ax = df.plot.hist(column=["fs"], by="layer", figsize=(10, 8))
    plt.savefig("data/dist_fs_" + name + '.png')
    plt.clf()
    ax = df.plot.hist(column=["qt"], by="layer", figsize=(10, 8))
    plt.savefig("data/dist_qt_" + name + '.png')
    plt.close()