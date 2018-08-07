import tensorflow as tf
import pandas as pd
import numpy as np
import logging

from sklearn.model_selection import train_test_split
import re
def load_imdb_sentiment_analysis_dataset( seed=123):
    train_file = './data/consumer_complaints.csv.zip'
    df = pd.read_csv(train_file, compression='zip', dtype={'consumer_complaint_narrative': object})
    selected = ['product', 'consumer_complaint_narrative']
    non_selected = list(set(df.columns) - set(selected))

    df = df.drop(non_selected, axis=1) # Drop non selected columns
    df = df.dropna(axis=0, how='any', subset=selected) # Drop null rows

    df.drop(df.tail(50000).index, inplace=True)
    df = df.reindex(np.random.permutation(df.index)) # Shuffle the dataframe
    x = df[selected[1]].apply(lambda x: clean_str(x)).tolist()
    y = df[selected[0]].apply(lambda x: clean_str(x)).tolist()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    """Step 3: shuffle the train set and split the train set into train and dev sets"""

    #x_train, x_dev, y_train, y_dev = train_test_split(x_, y_, test_size=0.1)

    print('x is ', x[0])
    print('y is ', y[0])
    print('x_test is ', x_test[0])
    print('y_test is ', y_test[0])
    print('x_train is ', x_train[0])
    print('y_train is ', y_train[0])

    print('Testing data completed')
    print('x: {}, x_train: {}, x_test: {}'.format(len(x), len(x_train), len(x_test)))
    print('y: {}, y_train: {}, y_test: {}'.format(len(y), len(y_train), len(y_test)))
    labels,unique_counts= np.unique(y,return_counts=True, axis=None)
    print('Labels',labels)
    print('uniquie_counts',unique_counts)
    print('Label \t counts')
    for i in range(len(unique_counts)):
        print(labels[i],unique_counts[i])

    np.random.seed(seed)
    np.random.shuffle(x_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)
    np.random.seed(seed)
    np.random.shuffle(x_test)
    np.random.seed(seed)
    np.random.shuffle(y_test)
    return((x_test,y_test),(x_train,y_train))


def clean_str(s):
    """Clean sentence"""
    s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
    s = re.sub(r"\'s", " \'s", s)
    s = re.sub(r"\'ve", " \'ve", s)
    s = re.sub(r"n\'t", " n\'t", s)
    s = re.sub(r"\'re", " \'re", s)
    s = re.sub(r"\'d", " \'d", s)
    s = re.sub(r"\'ll", " \'ll", s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " \( ", s)
    s = re.sub(r"\)", " \) ", s)
    s = re.sub(r"\?", " \? ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r'\S*(x{2,}|X{2,})\S*',"xxx", s)
    s = re.sub(r'[^\x00-\x7F]+', "", s)
    return s.strip().lower()

if __name__== "__main__":
    load_imdb_sentiment_analysis_dataset()