import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


class AutoEncoderDecoder:
    def __init__(self, shape, epochs=10, batch_size=20, shuffle=True):
        # settings
        self.shape = shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        # variables
        self.autoencoder = []
        self.prediction = []
        # execute encoder decoder
        self.__encoder_decoder()

    def __encoder_decoder(self):
        input = layers.Input(shape=self.shape)

        # build encoder
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
        # x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        # x = layers.MaxPooling2D((2, 2), padding="same")(x)

        # build decoder
        x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
        x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
        # x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
        x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

        self.autoencoder = Model(input, x)

    def compile_model(self, metrics="binary_crossentropy"):
        self.autoencoder.compile(optimizer="adam", loss=metrics)
        print(self.autoencoder.summary())

    def train(self, training_data, target_data, validation_data=None):
        self.autoencoder.fit(x=training_data,
                             y=target_data,
                             epochs=self.epochs,
                             batch_size=self.batch_size,
                             shuffle=self.shuffle,
                             # validation_data=validation_data,
                             )

    def predict(self, data):
        self.prediction = self.autoencoder.predict(data)
