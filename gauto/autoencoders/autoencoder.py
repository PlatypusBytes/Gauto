import sys
from tensorflow.keras import layers
from tensorflow.keras.models import Model
# own packages
from gauto.model_settings import ActivationFunctions, LossFunctions, Optimizer


class AutoEncoderDecoder:
    def __init__(self, shape, epochs=10, batch_size=20,
                 filters=[32, 64, 128], pooling=[2, 2, 2], kernel_size=3, shuffle=True, perc_val=0.8,
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
        self.percentage_val = perc_val  # percentage of validation. overwritten if data_validation exists
        # variables
        self.encoder = []
        self.decoder = []
        self.autoencoder = []
        self.prediction = []
        # execute encoder decoder
        self.__encoder_decoder()

    def __encoder_decoder(self):
        # build encoder
        incoded_input = layers.Input(shape=self.shape)
        encoder = incoded_input
        for i in range(self.nb_layers):
            encoder = layers.Conv2D(filters=self.filters[i],
                              kernel_size=(self.kernel_size, self.kernel_size),
                              activation=self.activation_fct,
                              padding="same")(encoder)
            encoder = layers.MaxPooling2D((self.pooling[i], self.pooling[i]),
                                    padding="same")(encoder)

        self.encoder = Model(incoded_input, encoder, name="Encoder")

        # build decoder
        decoded_input = layers.Input(shape=encoder.shape[1:])
        decoder = decoded_input
        for i in range(self.nb_layers-1, -1, -1):
            decoder = layers.Conv2DTranspose(filters=self.filters[i],
                                       kernel_size=(self.kernel_size, self.kernel_size),
                                       activation=self.activation_fct,
                                       padding="same")(decoder)
            decoder = layers.UpSampling2D((2, 2))(decoder)

        decoder = layers.Conv2D(filters=self.shape[2],
                          kernel_size=(self.kernel_size, self.kernel_size),
                          activation=self.activation_fct,
                          padding="same")(decoder)

        self.decoder = Model(decoded_input, decoder, name="Decoder")

    def compile_model(self):

        auto_input = layers.Input(shape=self.shape)
        enc = self.encoder(auto_input)
        dec = self.decoder(enc)

        self.autoencoder = Model(auto_input, dec, name="AutoEncoderDecoder")
        self.autoencoder.compile(optimizer=self.optimiser, loss=self.loss_fct)
        print(self.autoencoder.summary())

    def train(self, training_data, target_data, validation_data=None):
        self.autoencoder.fit(x=training_data,
                             y=target_data,
                             epochs=self.epochs,
                             batch_size=self.batch_size,
                             shuffle=self.shuffle,
                             validation_split=self.percentage_val,
                             validation_data=validation_data,
                             )
        # encoder
        self.encoder = Model(self.autoencoder.layers[1].input, self.autoencoder.layers[1].output)
        # decoder
        self.decoder = Model(self.autoencoder.layers[2].input, self.autoencoder.layers[2].output)

    def predict(self, data):
        self.prediction = self.autoencoder.predict(data)
