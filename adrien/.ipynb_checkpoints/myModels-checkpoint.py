#!/usr/bin/env python
# coding: utf-8

# In[1]:



import os

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Masking
from sklearn.preprocessing import normalize
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import multi_gpu_model
import math
from tensorflow import Graph, Session

from tensorflow import keras
import tensorflow as tf
import pickle
import numpy as np
# from tensorflow.python.client import device_lib
# from tensorflow.keras import backend as K
# from keras.utils import to_categorical

# print(device_lib.list_local_devices()) # list of DeviceAttributes

# %gui qt
import numpy as np
# import mne
import pickle
import sys
import os
import matplotlib

import matplotlib.pyplot as plt


# In[3]:


def dualLSTM(clas, sam, chans):


    model = Sequential()
#     model.add(Masking(mask_value=-9999, input_shape=(1250, test_X.shape[-1])))
#     model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99))

    model.add(tf.keras.layers.CuDNNLSTM(40, return_sequences=True, input_shape=(sam, chans)))
#     model.add(keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.CuDNNLSTM(25))
    model.add(keras.layers.Dropout(0.3))
#     model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.3))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(clas, activation='softmax'))
#     return model
    return model

# In[4]:


def singleLSTM(clas, sam, chans):

    model = Sequential()
#     model.add(Masking(mask_value=-9999, input_shape=(1250, test_X.shape[-1])))
#     model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99))

    model.add(tf.keras.layers.CuDNNLSTM(40, input_shape=(sam, chans)))
#     model.add(keras.layers.Dropout(0.1))
#     model.add(LSTM(25))
    model.add(keras.layers.Dropout(0.3))
#     model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.3))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(clas, activation='softmax'))
#     return model
    return model


# In[5]:





# In[ ]:




