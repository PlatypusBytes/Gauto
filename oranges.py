# gauto packages
from gauto.autoencoders.autoencoder import AutoEncoderDecoder
from gauto.autoencoders.utils import combine_encoders, load_figures, split_images, make_plot

# read images from ImageNet: https://www.image-net.org/index.php
oranges = load_figures(r"D:\imagenet-object-localization-challenge\imagenet_object_localization_patched2019.tar\Data\CLS-LOC\train",
                       "n07747607")
apples = load_figures(r"D:\imagenet-object-localization-challenge\imagenet_object_localization_patched2019.tar\Data\CLS-LOC\train",
                       "n07742313")

# split datasets
oranges_training, oranges_val = split_images(oranges, split_perc=0.8)
apples_training, apples_val = split_images(apples, split_perc=0.8)

# perform autoencoding / decoding => oranges
auto_oranges = AutoEncoderDecoder(oranges_training.shape[1:], filters=[64, 128, 256, 512], pooling=[2, 2, 2], epochs=150, batch_size=20)
auto_oranges.compile_model()
auto_oranges.train(oranges_training, oranges_training, validation_data=(oranges_val, oranges_val))
auto_oranges.predict(oranges_val)
for i, val in enumerate(oranges_val):
    make_plot([val, auto_oranges.prediction[i]], ["original", "autoencoders"], f"orange_{i}", output_f="./oranges")

# perform autoencoding / decoding => apples
auto_apples = AutoEncoderDecoder(apples_training.shape[1:], filters=[64, 128, 256, 512], pooling=[2, 2, 2], epochs=150, batch_size=20)
auto_apples.compile_model()
auto_apples.train(apples_training, apples_training, validation_data=(apples_val, apples_val))
auto_apples.predict(apples_val)
for i, val in enumerate(apples_val):
    make_plot([val, auto_apples.prediction[i]], ["original", "autoencoders"], f"apple_{i}", output_f="./apples")

# perform change of encoders
new = combine_encoders(apples_val, apples.encoder, oranges.decoder)
for i, val in enumerate(apples_val):
    make_plot([val, new[i]], ["original", "autoencoders"], f"apples2orange_{i}", output_f="./apples2orange")
