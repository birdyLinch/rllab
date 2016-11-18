import numpy as np

np.random.seed(1337) # for reproducibility

import theano
import theano.tensor as T

import numpy as np
import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,MaxoutDense
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
import matplotlib as mpl
mpl.use('TkAgg')
from keras.models import Model

import scipy.io as sio

class GAN():
    def __init__(fileName='MocapData.mat'):
        data=sio.loadmat(fileName)['data'][0]
        X = np.concatenate([np.asarray(frame) for frame in data],0)
        usedDim = np.ones(X.shape[1]).astype('bool')
        usedDim[39:45] = False
        X = X[:,usedDim]
        print(X.shape)
        self.mocap_pose = X
        self.discriminator=self.build_discreminator()
        self.discriminator_reward_scale = 0

    def build_discreminator(yShape=56, h=256):
        d_input = Input(shape=[yShape])
        H = Dense(2*h)(d_input)
        H = Activation('tanh')(H)
        H = Dense(2*h)(H)
        H = Activation('tanh')(H)
        H = Dense(2*h)(H)
        H = Activation('tanh')(H)
        H = Dense(1)(H)
        d = Activation('sigmoid')(H)
        discriminator = Model(d_input, d)
        discriminator.compile(loss='binary_crossentropy', optimizer=Adam())
        discriminator.summary()
        return discriminator

    def train(sample_pose, nb_epoch = 5000, batch_size=32):
        generated_poses = sample_pose 
        X = np.concatenate([self.mocap_pose, generated_poses],0)
        Y = np.zeros([x.shape[0],1])
        Y[:self.mocap_pose.shape[0]] = 1
        for t in range(nb_epoch):
            # Train the discriminator on generated pose    
            mask = np.random.randint(0,X.shape[0],size=batch_size)
            x = X[mask]
            y = Y[mask]
            d_loss = self.discriminator.train_on_batch(x,y)

        return d_loss

    def forward(single_pose):
        pose = self.transform(single_pose)
        return self.discriminator.predict(pose)

    def transform(single_pose):
        pass

    def get_reward(single_pose):
        return self.forward(single_pose)*self.discriminator_reward_scale

gan=GAN()

