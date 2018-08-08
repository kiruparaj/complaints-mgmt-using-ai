"""Module to train n-gram model.

Vectorizes training and validation texts into n-grams and uses that for
training a n-gram model - a simple multi-layer perceptron model. We use n-gram
model for text classification when the ratio of number of samples to number of
words per sample for the given dataset is very small (<~1500).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
#import time
from time import gmtime, strftime
import tensorflow as tf
import numpy as np

import build_model_kj
import load_data_ratna
import load_data_kj

import vectorize_data_kj
import explore_data_kj

FLAGS = None


def train_ngram_model(data,
                      learning_rate=1e-3,
                      epochs=1000,
                      batch_size=128,
                      layers=2,
                      units=64,
                      dropout_rate=0.2):
    """Trains n-gram model on the given dataset.

    # Arguments
        data: tuples of training and test texts and labels.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of Dense layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.

    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """
    # Get the data.
    print('==================== BEGINNING THE TRAINING ================================ ')
    (train_texts, train_labels), (val_texts, val_labels) = data

    # Verify that validation labels are in the same range as training labels.
    #num_classes = explore_data_kj.get_num_classes(train_labels)
    #print('printing num clsses: ',num_classes)
    num_classes = 11
   # unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
    #if len(unexpected_labels):
   #     raise ValueError('Unexpected label values found in the validation set:'
    #                     ' {unexpected_labels}. Please make sure that the '
    #                     'labels in the validation set are in the same range '
    #                     'as training labels.'.format(
   #                          unexpected_labels=unexpected_labels))
    print('==================== VECTORIZING THE TEXTS ================================ ')

    # Vectorize texts.
    train_labels=vectorize_data_kj.vectorize_labels(train_labels)
    val_labels = vectorize_data_kj.vectorize_labels(val_labels)

    x_train, x_val = vectorize_data_kj.ngram_vectorize(
        train_texts, train_labels, val_texts)

    print('==================== CREATING A MODEL INSTANCE ================================ ')

    # Create model instance.
    model = build_model_kj.mlp_model(layers=layers,
                                  units=units,
                                  dropout_rate=dropout_rate,
                                  input_shape=x_train.shape[1:],
                                  num_classes=num_classes)

    # Compile model with learning parameters.
    print('==================== COMPILING THE MODEL WITH LEARNING PARAMS ================================ ')

    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2)]

    # Train and validate model.
    print('==================== TRAINING AND VALIDATING THE MODEL ================================ ')
    print('x_train',x_train[0])
    print('train labels',train_labels[0])

    history = model.fit(
            x_train,
            train_labels,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(x_val, val_labels),
            verbose=2,  # Logs once per epoch.
            batch_size=batch_size)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # evaluate the model
   # scores = model.evaluate(X, Y, verbose=0)
   # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")

    # Save model.
    model_name='./models/compliants-mgmt-mlp_' +strftime("%d_%b_%Y_%H_%M_%S", gmtime())+'.h5'
    #model_name.append(model_name,str(strftime("%d_%b_%Y_%H_%M_%S", gmtime())))
    #model_name.append(model_name,'.h5')
    print(model_name)
    model.save(model_name)


    return history['val_acc'][-1], history['val_loss'][-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='input data directory')
    FLAGS, unparsed = parser.parse_known_args()

    # Using the IMDb movie reviews dataset to demonstrate training n-gram model
    #data = load_data_ratna.load_data_from_csv(FLAGS.data_dir)
    starttime = gmtime()
    print('start time ',strftime("%d_%b_%Y_%H_%M_%S",starttime ))
    data = load_data_kj.load_data_from_csv(123)
    print('data loaded and back to train ===============')
    train_ngram_model(data)
    endtime = gmtime()
    print('end time ',strftime("%d_%b_%Y_%H_%M_%S",endtime ))
    print('total time taken in mins ',(strftime("%M_%S",endtime-starttime)) )