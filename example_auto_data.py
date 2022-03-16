import os.path
import pickle
from time import time
import numpy as np
import matplotlib.pylab as plt
from gauto.autoencoders.autoencoder import AutoEncoderDecoder


def make_plot(data0, data1, data2, name, output_f):

    if not os.path.isdir(os.path.dirname(output_f)):
        os.makedirs(os.path.dirname(output_f))

    x = np.linspace(0, data1.shape[0], data1.shape[0])
    y = np.linspace(0, data1.shape[1], data1.shape[1])
    z = np.linspace(0, data1.shape[2], data1.shape[2])
    x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z)

    # and plot everything
    fig = plt.figure(figsize=(15, 5))
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax0.set_position([0.025, 0.15, 0.3, 0.7])
    ax1.set_position([0.325, 0.15, 0.3, 0.7])
    ax2.set_position([0.625, 0.15, 0.3, 0.7])
    ax0.view_init(45, 145)
    ax1.view_init(45, 145)
    ax2.view_init(45, 145)
    ax0.set_title("Exact")
    ax1.set_title("Noise")
    ax2.set_title("AutoEnconder")
    ax0.scatter(x_mesh, y_mesh, z_mesh, c=data0, vmin=np.min(data0), vmax=np.max(data0))
    ax1.scatter(x_mesh, y_mesh, z_mesh, c=data1, vmin=np.min(data0), vmax=np.max(data0))
    ax2.scatter(x_mesh, y_mesh, z_mesh, c=data2, vmin=np.min(data0), vmax=np.max(data0))
    ax0.set_xlabel("X dim", fontsize=10)
    ax1.set_xlabel("X dim", fontsize=10)
    ax2.set_xlabel("X dim", fontsize=10)
    ax0.set_ylabel("Y dim", fontsize=10)
    ax1.set_ylabel("Y dim", fontsize=10)
    ax2.set_ylabel("Y dim", fontsize=10)
    ax0.set_zlabel("Z dim", fontsize=10)
    ax1.set_zlabel("Z dim", fontsize=10)
    ax2.set_zlabel("Z dim", fontsize=10)

    mappable = plt.cm.ScalarMappable()
    mappable.set_array(data0)
    mappable.set_clim(np.min(data0), np.max(data0))
    cax = plt.axes([0.95, 0.20, 0.02, 0.5])
    cax.tick_params(labelsize=8)
    cbar = plt.colorbar(mappable, cax=cax, fraction=0.1, pad=0.01)
    cbar.set_label(name, fontsize=10)
    plt.savefig(f"{output_f}.png")
    # plt.show()
    plt.close()


# load data
with open(r"./data/qt.pickle", "rb") as fi:
    data_tr, data_val = pickle.load(fi)

# normalise data
for i in range(data_tr.shape[0]):
    data_tr[i] /= np.max(data_tr[i])

data_val_noise = np.copy(data_val)
for i in range(data_val.shape[0]):
    data_val[i] /= np.max(data_val[i])
    # add noise to validation
    data_val_noise[i] += 1 * np.random.normal(loc=0.0, scale=1.0, size=data_val_noise[i].shape)
    data_val_noise[i] /= np.max(data_val_noise[i])

# perform autoencoding / decoding
t_ini = time()
auto = AutoEncoderDecoder(data_tr.shape[1:], filters=[64, 128, 256], pooling=[2, 2, 2], epochs=150, batch_size=20)
auto.compile_model()
auto.train(data_tr, data_tr)
auto.predict(data_val)
print(f"Elapsed time: {time() - t_ini} s")

auto.predict(data_val_noise)

for i, val in enumerate(data_val_noise):
    make_plot(data_val[i], val, auto.prediction[i], "value", f"./autoencoder_noise/auto_encoder_{i}")
