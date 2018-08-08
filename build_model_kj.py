"""Module to create model.

Helper functions to create a multi-layer perceptron model and a separable CNN
model. These functions take the model hyper-parameters as input. This will
allow us to create model instances with slightly varying architectures.
"""
from tensorflow.python.keras import models

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout

import numpy as np

def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    """Creates an instance of a multi-layer perceptron model.

    # Arguments
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.

    # Returns
        An MLP model instance.
    """
    print('==================== BUILDING THE MLP MODEL ================================ ')

    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    print('============================ UNITS: ', units)
    print('============================ DROPOUT RATE: ', dropout_rate)
    print('============================ LAYERS: ', layers)
    print('============================ INPUT SHAPE: ', input_shape)
    print('============================ NUM CLASSES: ', num_classes)
    for _ in range(layers-1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))
        print('==================IN LOOP ========== DROPOUT RATE: ', dropout_rate)

    #model.add(Flatten())
    print('============================ OP UNITS: ', op_units)
    print('============================ OP ACTIVATION: ', op_activation)
    model.add(Dense(units=op_units, activation=op_activation))
    print(model)
    return model


def _get_last_layer_units_and_activation(num_classes):
    """Gets the # units and activation function for the last network layer.

    # Arguments
        num_classes: int, number of classes.

    # Returns
        units, activation values.
    """
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation

