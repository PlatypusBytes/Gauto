import os
import cv2
import imageio
import numpy as np
import matplotlib.pylab as plt


def combine_encoders(data, encoder, decoder):
    aux = encoder.predict(data)
    aux = decoder.predict(aux)
    return aux


def make_plot(data, title, name, x_label=None, y_label=None, output_f="./"):

    if not os.path.isdir(output_f):
        os.makedirs(output_f)

    nb_data = len(data)

    # and plot everything
    fig, ax = plt.subplots(nb_data, 1, figsize=(5*nb_data, 5))
    for i in range(nb_data):
        ax[i].set_position([0.05 * (i + 1) + 0.3 * i, 0.15, 0.3, 0.7])
        ax[i].set_title(title[i])
        ax[i].imshow(data[i].astype(np.uint8))
        if x_label != None:
            ax[i].set_xlabel(x_label[i], fontsize=10)
        if y_label != None:
            ax[i].set_xlabel(y_label[i], fontsize=10)

    plt.savefig(f"{os.path.join(output_f, name)}.png")
    plt.close()


def load_figures(path_to_data, data_class, resize=(128, 128), n_dim=3):

    # list all files in path_to_data
    files = os.listdir(os.path.join(path_to_data, data_class))

    # read data class
    data = np.empty((len(files), resize[0], resize[1], n_dim))

    # for all files read
    for i, f in enumerate(files):
        img = imageio.imread(os.path.join(path_to_data, data_class, f))
        if len(img.shape) != n_dim:
            continue
        data[i] = cv2.resize(img, resize)
    return data


def split_images(data, split_perc=0.8):

    # number realisations
    nb_realisations = len(data)
    # determine indexes for training
    nb_samples = int(nb_realisations * split_perc)
    # indexes for training
    idx_tra = np.random.choice(nb_realisations, size=nb_samples, replace=False)
    # indexes for validation
    idx_val = list(set(range(nb_realisations)) - set(idx_tra))

    return data[idx_tra], data[idx_val]
