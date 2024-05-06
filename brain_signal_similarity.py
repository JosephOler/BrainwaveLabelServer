# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:30:48 2024

@author: buing
"""
# brain_signal_similarity.py

import numpy as np
# import tensorflow as tf
# from keras.models import Model   
# from keras.layers import * 
from keras.layers import Input, Subtract, Multiply, Concatenate, Dense, BatchNormalization, Dropout, Activation
from keras.models import Model
# from tensorflow import keras
# from tensorflow.keras.layers import *
# from tensorflow.keras import layers, models
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

# import tensorflow as tf
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.applications.vgg16 import preprocess_input

# Function to compute Pearson correlation coefficient
def correlation(signal1, signal2):
    """
    Computes the Pearson correlation coefficient between two signals.
    """
    return pearsonr(signal1, signal2)[0]

# Function to compute Euclidean distance
def euclidean_distance(signal1, signal2):
    """
    Computes the Euclidean distance between two signals.
    """
    return np.linalg.norm(signal1 - signal2)

# Function to compute cosine similarity
def cosine_similarity(signal1, signal2):
    """
    Computes the cosine similarity between two signals.
    """
    dot_product = np.dot(signal1, signal2)
    magnitude1 = np.linalg.norm(signal1)
    magnitude2 = np.linalg.norm(signal2)
    return dot_product / (magnitude1 * magnitude2)


def Siamese_Model(features):
    """
    Implementation of the Siamese Network
    Args:
    features (int): number of features 
    Returns:
    [keras model]: siamese model
    """

    inp1 = Input(shape=(features,))
    inp2 = Input(shape=(features,))

    diff = Subtract()([inp1, inp2])
    # squared difference
    L2 = Multiply()([diff, diff])
    # product proximity
    prod = Multiply()([inp1, inp2])
    # combined metric
    combine = Concatenate(axis=1)([L2, prod])

    path1 = Dense(64)(L2)
    path1 = BatchNormalization()(path1)
    path1 = Dropout(0.25)(path1)
    path1 = Activation('relu')(path1)

    path2 = Dense(64)(prod)
    path2 = BatchNormalization()(path2)
    path2 = Dropout(0.25)(path2)
    path2 = Activation('relu')(path2)

    path3 = Dense(64)(combine)
    path3 = BatchNormalization()(path3)
    path3 = Dropout(0.25)(path3)
    path3 = Activation('relu')(path3)

    # combining everything
    paths = Concatenate(axis=1)([path1, path2, path3])

    top = Dense(256)(paths)
    top = BatchNormalization()(top)
    top = Dropout(0.25)(top)
    top = Activation('relu')(top)

    out = Dense(1)(top)   # output similarity score

    siamese_model = Model(inputs=[inp1, inp2], outputs=[out])

    return siamese_model
