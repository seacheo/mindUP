#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("/home/sean/pench")
sys.path.append("/network/lustre/iss01/home/adrien.martel")
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# !git clone https://github.com/vlawhern/arl-eegmodels.git

from eegmodels.EEGModels import EEGNet, ShallowConvNet, DeepConvNet
from myModels import dualLSTM, singleLSTM
import tensorflow as tf
from tensorflow import keras
tf.enable_eager_execution()
from threading import Thread

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import normalize

import math
import threading


import pickle
import numpy as np
from tensorflow.python.client import device_lib
from tensorflow.keras.utils import to_categorical
# import keras
# from tqdm.keras import TqdmCallback

print(device_lib.list_local_devices()) # list of DeviceAttributes

# %gui qt
import numpy as np
# import mne
import pickle
import os
import matplotlib

import matplotlib.pyplot as plt
from multiprocessing import Pool, Queue
import multiprocessing
# tf.enable_eager_execution()
from collections import deque

from tensorflow.keras.backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)


# In[2]:


def randomize(a, b, c):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    shuffled_c = c[permutation]
    return shuffled_a, shuffled_b, shuffled_c


# In[3]:


baseFolder='one/'
baseFolder='/network/lustre/iss01/home/adrien.martel/data/MW/'
files=[f for f in os.listdir(baseFolder) if not f.startswith('.')]


# In[4]:


def createData(file):
    data=pickle.load(open(baseFolder+file, 'rb'))
    
    sfreq=512
    features=[]
    flipFeatures=[]
    labels=[]
    for i in range(numClasses):
        for k in range(len(data[i])):
            labels.append(i)
            features.append(data[i][k])
            flipFeatures.append([np.transpose(data[i][k])])
    labels=np.array(labels)
    features=np.array(features)
    flipFeatures=np.array(flipFeatures)
    labels, features, flipFeatures = randomize(labels, features, flipFeatures)
    
    labels = to_categorical(labels, num_classes=numClasses)
    return [features, flipFeatures, labels]


# In[5]:


sam=2560
chans=62
numClasses=2


# In[6]:


models  = [
    [EEGNet(nb_classes=numClasses, Chans=chans, Samples=sam), True, 'EEGNet-V1'], 
    [ShallowConvNet(nb_classes=numClasses, Chans=chans, Samples =sam), True, 'ShallowConvNet-V1'], 
    [DeepConvNet(nb_classes=numClasses, Chans=chans, Samples=sam), True, 'DeepConvNet-V1'],
    [singleLSTM(clas=numClasses, sam=sam, chans=chans), False, 'singleLSTM-V1'],
    [dualLSTM(clas=numClasses, sam=sam, chans=chans), False, 'dualLSTM-V1'],
    ]


# In[7]:


def createWork(n):
    arc=inps[n][0]
    file=inps[n][1]
    features, flipFeatures, labels = createData(file)
    if arc[1]:
        train_X = np.array(flipFeatures[0:int(7*len(labels)/10)])
        test_X = np.array(flipFeatures[int(7*len(labels)/10):-1])
    else:
        train_X = np.array(features[0:int(7*len(labels)/10)])
        test_X = np.array(features[int(7*len(labels)/10):-1])
    train_y = np.array(labels[0:int(7*len(labels)/10)])
    test_y = np.array(labels[int(7*len(labels)/10):-1])
    print("Putted", file, out.empty())
#     out.put([arc[0], train_X, test_X, train_y, test_y, file, arc[2]])
    print(file, train_X.shape, test_X.shape, train_y.shape, test_y.shape)
#     print(out.empty())
    return [arc[0], train_X, test_X, train_y, test_y, file, arc[2]]


# In[8]:


inps=[]
for model in models:
    try:
        os.mkdir(model[2])
    except:
        print("probably exists")
    for file in files:
        inps.append([model, file])


# In[9]:


manager = multiprocessing.Manager()
out = manager.Queue()


# In[10]:


# p = Pool(20)
# master=p.map(createWork, list(range(2)))
# master=p.map(createWork, list(range(len(inps))))


# In[11]:


gpus=4
out.empty()


# In[12]:


# out = Queue()
# out.queue = queue.deque(master)
[out.put(i) for i in list(range(len(inps)))]


# In[13]:


out.empty()


# In[14]:


def doWork(i):
#     i = args[0]
#     out = args[1]
#     i=1
    os.environ["CUDA_VISIBLE_DEVICES"]=str(i)
    while not out.empty():
#         dat=out.get()
#         print("not empty")
        n=out.get()
        dat= createWork(n)
        model=dat[0]
        train_X=dat[1]
        test_X=dat[2]
        train_y=dat[3]
        test_y=dat[4]
        file=dat[5]
        folder=dat[6]
#         print('processed')
#         sgd = keras.optimizers.SGD(learning_rate=0.015, momentum=0.0, nesterov=False)
#         adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        print('Done getting data')
#         sgd = keras.optimizers.SGD()
        adam = tf.train.AdamOptimizer(
    learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
    name='Adam'
)
        print('Compiling model')
#         break
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        # fit network

        history = model.fit(train_X, train_y, epochs=10, batch_size=2, validation_data=(test_X, test_y), verbose=0, shuffle=True)
        # plot history
        print(history.history.keys())
        pyplot.figure(figsize=(25,10), dpi=250)
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.plot(history.history['accuracy'], label='accuracy')
        pyplot.plot(history.history['val_accuracy'], label='test accuracy')
        pyplot.legend()
        pyplot.savefig(folder+'/'+file + '.png')

        pickle.dump(history, open(folder+'/'+file+'-hist.p', "wb"))
        model.save(folder+'/'+file+'.h5')
    print('done')


# In[15]:


workers=[]
for i in range(gpus):
    
    workers.append(Thread(target = doWork, args=(i,)))
for worker in workers:
    worker.start()  


# In[16]:


# s = Pool(2)

# master=s.map(doWork, [(x, out) for x in range(gpus)])


# In[17]:


worker.join()


# In[ ]:




