import numpy as np
from gstools import SRF, Exponential, Gaussian, Matern, Linear
import matplotlib.pyplot as plt
import pickle


def get_srf(model_name, value_name, mean, var, seed=1, len_scale=[10, 10, 0.5], angles=0.5):

    ndim = 3

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
        print('model name: "', value_name, '" is not supported')

    return SRF(model, mean=mean, seed=seed)


def make_plot(x, y, z, data, name):

    x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z)

    # and plot everything
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection='3d')
    ax.set_position([0.1, 0.15, 0.5, 0.7])
    ax.view_init(45, 145)
    ax.scatter(x_mesh, y_mesh, z_mesh, c=data)
    ax.set_xlabel("X dim", fontsize=10)
    ax.set_ylabel("Y dim", fontsize=10)
    ax.set_zlabel("Z dim", fontsize=10)

    mappable = plt.cm.ScalarMappable()
    mappable.set_array(data)
    # mappable.set_clim(v_min, v_max)
    cax = plt.axes([0.77, 0.20, 0.02, 0.5])
    cax.tick_params(labelsize=8)
    cbar = plt.colorbar(mappable, cax=cax, fraction=0.1, pad=0.01)
    cbar.set_label(name, fontsize=10)
    plt.savefig(f"{name}.png")
    plt.show()
    plt.close()


def dump_data(data, name):
    with open(name, "wb") as fo:
        pickle.dump(data, fo)


if __name__ == "__main__":
    nb_realisations = 1000
    split = 0.8

    x = np.linspace(0, 30, 32)
    y = np.linspace(0, 30, 32)
    z = np.linspace(0, -10, 12)
    qt_data_training = []
    qt_data_validation = []

    # determine indexes for training
    nb_samples = int(nb_realisations * split)
    # indexes for validation
    idx_tr = np.random.choice(nb_realisations, size=nb_samples, replace=False)

    for i in range(nb_realisations):
        print(i)
        qt = get_srf("Exponential", "qt", 5, 2, seed=i)
        qt = qt((x, y, z), mesh_type='structured')
        if i in idx_tr:
            qt_data_training.append(qt)
        else:
            qt_data_validation.append(qt)

    qt_data_training = np.array(qt_data_training)
    qt_data_validation = np.array(qt_data_validation)

    make_plot(x, y, z, qt_data_training[0], "qt")
    dump_data([qt_data_training, qt_data_validation], "qt.pickle")

    with open("qt.pickle", "rb") as fi:
        data_tr, data_val = pickle.load(fi)
    plt.plot(data_tr[0, 5, 5, :], z)
    plt.show()
