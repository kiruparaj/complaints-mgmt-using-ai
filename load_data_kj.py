"""Module to load data.

Consists of functions to load data from a CSV do the following:
    - Read the required fields (texts and labels).
    - Do any pre-processing if required. For example, make sure all label
        values are in range [0, num_classes-1].
    - Split the data into training and validation sets.
    - Shuffle the training data.
"""
import os
import pandas as pd
import numpy as np
import sys
from time import gmtime, strftime
import argparse

FLAGS = None

def load_data_from_csv (data_path,
                        validation_split=0.2,
                        seed=123):
    """Loads the mortgage customer complaint dataset.

        # Arguments
            data_path: string, path to the data directory.
            validation_split: float, percentage of data to use for validation.
            seed: int, seed for randomizer.

        # Returns
            A tuple of training and validation data.
            Number of training samples: 
            Number of test samples: 
            Number of categories:  

        # References
            https://www.kaggle.com/tmorrison/mortgage-complaints/data

            Download and uncompress archive from:
            https://www.kaggle.com/cfpb/us-consumer-finance-complaints/downloads/consumer_complaints.csv/1
        """
    print('load_data_from_csv is called: ',data_path)
    columns = (1, 5)  # 1 - product, 5 - consumer_complaint_narrative.
    data = _load_and_shuffle_data(data_path,'consumer_complaints.csv', columns, seed)
    print('data is loaded')
    print(len(data[0][0]),len(data[1][0]))

    return data

def _load_and_shuffle_data(data_path,
                           file_name,
                           cols,
                           seed,
                           separator=',',
                           header=0):
    """Loads and shuffles the dataset using pandas.

    # Arguments
        data_path: string, path to the data directory.
        file_name: string, name of the data file.
        cols: list, columns to load from the data file.
        seed: int, seed for randomizer.
        separator: string, separator to use for splitting data.
        header: int, row to use as data header.
    """
    print('_load_and_shuffle_data is called: file_name: step 1 ', data_path, file_name)
    np.random.seed(seed)
    #data_path = os.path.join(data_path, file_name)
    data_path = './data/consumer_complaints.csv.zip'
    print('_load_and_shuffle_data is called: data_path:', data_path)
    data = pd.read_csv(data_path, compression='zip',usecols=cols, sep=separator)
    data = data.dropna(axis=0, how='any')
    print('column names', data.columns)
    #data = pd.read_csv(data_path, compression='zip', dtype={'consumer_complaint_narrative': object})
    data = data.reindex(np.random.permutation(data.index))
    texts = list(data['consumer_complaint_narrative'])
    labels = np.array(data['product'])
    print('length of texts', len(texts))
    print('length of ccn',len(labels))
    return _split_training_and_validation_sets(texts, labels, .2)


def _split_training_and_validation_sets(texts, labels, validation_split):
    """Splits the texts and labels into training and validation sets.

    # Arguments
        texts: list, text data.
        labels: list, label data.
        validation_split: float, percentage of data to use for validation.

    # Returns
        A tuple of training and validation data.
    """
    num_training_samples = int((1 - validation_split) * len(texts))
    return ((texts[:num_training_samples], labels[:num_training_samples]),
            (texts[num_training_samples:], labels[num_training_samples:]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='input data directory')
    FLAGS, unparsed = parser.parse_known_args()

    # Using the IMDb movie reviews dataset to demonstrate training n-gram model
    data = load_data_from_csv(FLAGS.data_dir)
