from preprocess_utils import data_loader, data_loader_with_coords

import tensorflow as tf

import os
import time
import datetime
import numpy as np
import pandas as pd
import random
import plotly.express as px

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from IPython import display


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def downsample3d(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv3D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result


def upsample3D(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv3DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result


def Generator3d(vol_rows=64, vol_cols=64, vol_height=64, channels=1, output_channels=1):
    vol_shape = (vol_rows, vol_cols, vol_height, channels)
    inputs = tf.keras.layers.Input(shape=vol_shape)
    down_stack = [
        downsample3d(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample3d(128, 4),  # (batch_size, 64, 64, 128)
        downsample3d(256, 4),  # (batch_size, 32, 32, 256)
        downsample3d(512, 4),  # (batch_size, 16, 16, 512)
    ]
    up_stack = [
        upsample3D(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample3D(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample3D(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample3D(512, 4),  # (batch_size, 16, 16, 1024)
        upsample3D(256, 4),  # (batch_size, 32, 32, 512)
        upsample3D(128, 4),  # (batch_size, 64, 64, 256)
        upsample3D(64, 4),  # (batch_size, 128, 128, 128)
    ]
    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = tf.keras.layers.Conv3DTranspose(
        output_channels,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
    )  # (batch_size, 256, 256, 3)
    x = inputs
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 1])
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]
    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        OUTPUT_CHANNELS,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
    )  # (batch_size, 256, 256, 3)
    x = inputs
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


def Discriminator3D(
    vol_rows=256, vol_cols=256, vol_height=256, channels=1, output_channels=1
):
    vol_shape = (vol_rows, vol_cols, vol_height, channels)
    initializer = tf.random_normal_initializer(0.0, 0.02)
    inp = tf.keras.layers.Input(shape=vol_shape, name="input_image")
    tar = tf.keras.layers.Input(shape=vol_shape, name="target_image")
    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)
    down1 = downsample3d(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample3d(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample3d(256, 4)(down2)  # (batch_size, 32, 32, 256)
    zero_pad1 = tf.keras.layers.ZeroPadding3D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv3D(
        512, 4, strides=1, kernel_initializer=initializer, use_bias=False
    )(
        zero_pad1
    )  # (batch_size, 31, 31, 512)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding3D()(leaky_relu)  # (batch_size, 33, 33, 512)
    last = tf.keras.layers.Conv3D(1, 4, strides=1, kernel_initializer=initializer)(
        zero_pad2
    )  # (batch_size, 30, 30, 1)
    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def Discriminator():
    initializer = tf.random_normal_initializer(0.0, 0.02)
    inp = tf.keras.layers.Input(shape=[256, 256, 1], name="input_image")
    tar = tf.keras.layers.Input(shape=[256, 256, 1], name="target_image")
    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)
    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer, use_bias=False
    )(
        zero_pad1
    )  # (batch_size, 31, 31, 512)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)
    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(
        zero_pad2
    )  # (batch_size, 30, 30, 1)
    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(
        tf.zeros_like(disc_generated_output), disc_generated_output
    )
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target
        )
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    generator_gradients = gen_tape.gradient(
        gen_total_loss, generator.trainable_variables
    )
    discriminator_gradients = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )
    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )
    with summary_writer.as_default():
        tf.summary.scalar("gen_total_loss", gen_total_loss, step=step // 1000)
        tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=step // 1000)
        tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=step // 1000)
        tf.summary.scalar("disc_loss", disc_loss, step=step // 1000)


def generate_images_3d(model, test_input, tar, coords, savefig=True, step=None):
    
    prediction = model(test_input, training=True)
    test_input = np.array([slice_t.T for slice_t in test_input.numpy()[0, ..., 0]])
    tar = np.array([slice_t.T for slice_t in tar.numpy()[0, ..., 0]])
    prediction = np.array([slice_t.T for slice_t in prediction.numpy()[0, ..., 0]])
    display_list = np.array([test_input, tar, prediction])
    fig = px.imshow(display_list , animation_frame=1, facet_col=0)
    title = ["Input Image", "Ground Truth", "Predicted Image"]
    for counter, annotation in enumerate(fig.layout.annotations):
        annotation.text = title[counter]
    fig.update_layout(sliders=[{"currentvalue": {"prefix": "No of slice in the y direction="}}])
    if savefig:
        fig.write_html(f"gauto\\gans\\pix2pix\\output\\train_pix2pix{step}.html")
    else:
        fig.show()



def generate_images(model, test_input, tar, savefig=True, step=None):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ["Input Image", "Ground Truth", "Predicted Image"]
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis("off")
    if savefig:
        plt.savefig(f"gauto\\gans\\pix2pix\\output\\train_pix2pix{step}.png")
    else:
        plt.show()
    plt.clf()
    plt.hist(np.array(tar[0]).flatten(), bins=100, alpha=0.5, label="Ground Truth")
    plt.hist(np.array(prediction[0]).flatten(), bins=100, alpha=0.5, label="Prediction")
    plt.legend(loc="upper right")
    plt.title(f"Results at step {step}")
    if savefig:
        plt.savefig(f"gauto\\gans\\pix2pix\\output\\train_pix2pix_dist{step}.png")
    else:
        plt.show()


def generate_images_slider(model, training_dataset):
    dataset = [value for counter, value in enumerate(training_dataset)]
    prediction = model(dataset[0][0], training=True)
    fig, ax = plt.subplots(1, 3)
    fig.subplots_adjust(bottom=0.35)
    display_list = [dataset[0][0][0], dataset[0][1][0], prediction[0]]
    title = ["Input Image", "Ground Truth", "Predicted Image"]
    for i in range(3):
        ax[i].set_title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        ax[i].imshow(display_list[i] * 0.5 + 0.5)
        ax[i].axis("off")

    axslice = plt.axes([0.25, 0.15, 0.65, 0.03])
    freq = Slider(axslice, "Slice number", 0, len(dataset), 0, valstep=1)

    def update(val):
        prediction = model(dataset[freq.val][0], training=True)
        display_list = [dataset[freq.val][0][0], dataset[freq.val][1][0], prediction[0]]
        title = ["Input Image", "Ground Truth", "Predicted Image"]
        for i in range(3):
            ax[i].set_title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            ax[i].imshow(display_list[i] * 0.5 + 0.5)
            ax[i].axis("off")

    freq.on_changed(update)
    plt.show()


def fit(train_ds, test_ds, steps, generator, discriminator,  is_3D=False, coords=None):
    example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()
    for step, value_to_unpack in train_ds.repeat().take(steps).enumerate():
        (input_image, target) = value_to_unpack
        if (step) % 1000 == 0:
            display.clear_output(wait=True)
            if step != 0:
                print(f"Time taken for 1000 steps: {time.time() - start:.2f} sec\n")
            start = time.time()
            if is_3D:
                generate_images_3d(generator, example_input, example_target, step=step, coords=coords)
            else:
                generate_images(generator, example_input, example_target, step=step)
            print(f"Step: {step // 1000}k")
        loss_object = tf.keras.losses.MeanSquaredError()
        train_step(input_image, target, step)
        # Training step
        if (step + 1) % 10 == 0:
            print(".", end="", flush=True)
            # Save (checkpoint) the model every 5k steps
        if (step + 1) % 5000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

def get_csv_files_names(directory):
    directory = os.path.join(directory)
    rfs = []
    for root,dirs,files in os.walk(directory):
        for file in files:
           if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(directory,file), delimiter=',')
                rfs.append(df)
    return rfs

def load_data_and_normalize(name_dataset):
    MAX_QT = 10.5
    MAX_FS = 111.5
    full_data, train_input, binary_missing_data = data_loader(
        name_dataset, 0.9, "qt", sample_vertically=False
    )
    # define input shape based on the loaded dataset
    dataset = []
    dataset.append(np.reshape(train_input, (100, 256, 256, 1)))
    full_data, train_output, binary_missing_data = data_loader(
        name_dataset, 0, "fs", sample_vertically=False
    )
    dataset.append(np.reshape(train_output, (100, 256, 256, 1)))

    image_shape = dataset[0].shape[1:]
    dataset[0] = (dataset[0] / MAX_QT) - 1
    dataset[1] = (dataset[1] / MAX_FS) - 1

    return np.array(dataset).astype(np.float32)


def load_data_and_normalize_single_value(name_dataset):
    train_output, train_input, binary_missing_data = data_loader(
        name_dataset, 0.9, "IC", sample_vertically=True
    )
    # define input shape based on the loaded dataset
    dataset = []
    dataset.append(np.reshape(train_input, (100, 256, 256, 1)))
    dataset.append(np.reshape(train_output, (100, 256, 256, 1)))

    image_shape = dataset[0].shape[1:]
    dataset[0] = dataset[0] / np.amax(train_output)
    dataset[1] = dataset[1] / np.amax(train_output)

    return np.array(dataset).astype(np.float32)


def load_data_and_normalize_3d_figure(
    name_dataset, artifial_cpts, resize_shape=(256, 256, 256, 1)
):
    data_x = []
    x, y, z = [], [], []
    grouped = name_dataset.groupby("x")
    for name, group in grouped:
        data_x.append(list(group['IC']))
        x.append(list(group["x"]))
        y.append(list(group["y"]))
        z.append(list(group["z"]))
    data_x = np.array(data_x, dtype=float)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)

    data_x = np.reshape(data_x, resize_shape[:-1])
    idxs = np.random.choice(resize_shape[0] * resize_shape[1], size=artifial_cpts, replace=0)
    random_miss = np.column_stack(np.unravel_index(idxs,(resize_shape[0] , resize_shape[1])))

    data_miss_array = np.zeros_like(data_x)
    for data_miss in random_miss:
        data_miss_array[data_miss[0], data_miss[1]] = data_x[data_miss[0], data_miss[1]]

     
    # define input shape based on the loaded dataset
    dataset = []
    dataset.append(np.reshape(data_miss_array, resize_shape))
    dataset.append(np.reshape(data_x, resize_shape))
    

    image_shape = dataset[0].shape[1:]
    dataset[0] = dataset[0] / AMAX
    dataset[1] = dataset[1] / AMAX

    return np.array(dataset).astype(np.float32)


def train_and_set_up_3d_model():
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    files = get_csv_files_names("D:\\gauto\\data\\cond_rf\\train")
    dataset = np.array([load_data_and_normalize_3d_figure(file_pd, 20, (64, 64, 64, 1)) for file_pd in files])

    input_dataset = tf.convert_to_tensor(tf.constant(dataset[:, 0, ...]))
    train_input_dataset = tf.data.Dataset.from_tensor_slices(input_dataset)
    train_input_dataset = train_input_dataset.batch(BATCH_SIZE)
    target_dataset = tf.convert_to_tensor(tf.constant(dataset[:, 1, ...]))
    train_target_dataset = tf.data.Dataset.from_tensor_slices(target_dataset)
    train_target_dataset = train_target_dataset.batch(BATCH_SIZE)
    train_dataset = tf.data.Dataset.zip((train_input_dataset, train_target_dataset))


    files_test = get_csv_files_names("D:\\gauto\\data\\cond_rf\\test")
    test_dataset = np.array([load_data_and_normalize_3d_figure(file_pd, 20, (64, 64, 64, 1)) for file_pd in files_test])

    input_dataset_test = tf.convert_to_tensor(tf.constant(test_dataset[:, 0, ...]))
    test_input_dataset = tf.data.Dataset.from_tensor_slices(input_dataset_test)
    test_input_dataset = test_input_dataset.batch(BATCH_SIZE)
    target_dataset_test = tf.convert_to_tensor(tf.constant(test_dataset[:, 1, ...]))
    test_target_dataset = tf.data.Dataset.from_tensor_slices(target_dataset_test)
    test_target_dataset = test_target_dataset.batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.zip((test_input_dataset, test_target_dataset))

    log_dir = "logs3d/"

    summary_writer = tf.summary.create_file_writer(
        log_dir + "fit_train3d/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    # test image generation
    # for example_input, example_target in test_dataset.take(1):
    #  generate_images(generator, example_input, example_target)
    checkpoint_dir = "./training_checkpoints3D"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )

    # create coordinates 
    x_grid = np.linspace(0, 64, 64)
    y_grid = np.linspace(0, 64, 64)
    z_grid = np.linspace(0, 64, 64)
    xs, ys, zs = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")

    fit(train_dataset, test_dataset, steps=1000, generator=generator, discriminator=discriminator,  is_3D=True, coords=np.array([xs, ys, zs]))
    print(1)


if __name__ == "__main__":
    # The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
    BATCH_SIZE = 1
    AMAX = 4.5
    OUTPUT_CHANNELS = 1
    LAMBDA = 100

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf


    # define optimizers
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    log_dir = "logs/"

    summary_writer = tf.summary.create_file_writer(
        log_dir + "fit_train/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    loss_object = tf.keras.losses.MeanSquaredError()
    generator = Generator3d()
    discriminator = Discriminator3D(
        vol_rows=64, vol_cols=64, vol_height=64, channels=1, output_channels=1
    )


    train_and_set_up_3d_model()
