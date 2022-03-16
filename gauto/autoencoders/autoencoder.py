import sys
from tensorflow.keras import layers
from tensorflow.keras.models import Model
# own packages
from gauto.model_settings import ActivationFunctions, LossFunctions, Optimizer


class AutoEncoderDecoder:
    def __init__(self, shape, epochs=10, batch_size=20,
                 filters=[32, 64, 128], pooling=[2, 2, 2], kernel_size=3, shuffle=True,
                 activation_fct="relu", loss_fct="mean_squared_error", optimiser="Adam"):

        if len(filters) != len(pooling):
            sys.exit(f"Length of filters ({len(filters)}) must"
                     f" be the same as length of pooling ({len(pooling)})")

        if activation_fct not in ActivationFunctions.values:
            sys.exit(f"Activation function {activation_fct} not valid.\n"
                     f"Needs to be: {ActivationFunctions.values}")
        if loss_fct not in LossFunctions.values:
            sys.exit(f"Loss function {loss_fct} not valid.\n"
                     f"Needs to be: {LossFunctions.values}")
        if optimiser not in Optimizer.values:
            sys.exit(f"Optimiser {optimiser} not valid.\n"
                     f"Needs to be: {Optimizer.values}")
        # settings
        self.shape = shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.filters = filters
        self.pooling = pooling
        self.kernel_size = kernel_size
        self.nb_layers = len(filters)
        self.activation_fct = activation_fct
        self.loss_fct = loss_fct
        self.optimiser = optimiser
        # variables
        self.autoencoder = []
        self.prediction = []
        # execute encoder decoder
        self.__encoder_decoder()

    def __encoder_decoder(self):
        input = layers.Input(shape=self.shape)

        # build encoder
        x = input
        for i in range(self.nb_layers):
            x = layers.Conv2D(filters=self.filters[i],
                              kernel_size=(self.kernel_size, self.kernel_size),
                              activation=self.activation_fct,
                              padding="same")(x)
            x = layers.MaxPooling2D((self.pooling[i], self.pooling[i]),
                                    padding="same")(x)

        # build decoder
        for i in range(self.nb_layers-1, -1, -1):
            x = layers.Conv2DTranspose(filters=self.filters[i],
                                       kernel_size=(self.kernel_size, self.kernel_size),
                                       activation=self.activation_fct,
                                       padding="same")(x)
            x = layers.UpSampling2D((2, 2))(x)

        x = layers.Conv2D(filters=self.shape[2],
                          kernel_size=(self.kernel_size, self.kernel_size),
                          activation=self.activation_fct,
                          padding="same")(x)

        self.autoencoder = Model(input, x)

    def compile_model(self):
        self.autoencoder.compile(optimizer=self.optimiser, loss=self.loss_fct)
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
