class Optimizer:
    values = ["SGD",
              "RMSprop",
              "Adam",
              "Adadelta",
              "Adagrad",
              "Adamax",
              "Nadam",
              "Ftrl",
              ]


class ActivationFunctions:
    values = ["relu",
              "sigmoid",
              "softmax",
              "softplus",
              "softsign",
              "tanh",
              "selu",
              "elu",
              ]


class LossFunctions:
    values =["binary_crossentropy",
             "categorical_crossentropy",
             "sparse_categorical_crossentropy",
             "mean_squared_error",
             "mean_absolute_error",
             "mean_squared_logarithmic_error",
             ]
