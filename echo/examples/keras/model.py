import sys 
import yaml
import math
import time
import random
import traceback
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt
import scipy.sparse
from scipy.ndimage import gaussian_filter

from tqdm.auto import tqdm

import numpy.fft as FFT
from typing import List, Dict

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from tensorflow.keras.layers import (Input, Conv2D, Dense, Flatten, 
                                     MaxPool2D, RepeatVector, Lambda,
                                     LeakyReLU, Dropout)
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.backend as K

from keras_radam import RAdam
from keras_radam.training import RAdamOptimizer



class Conv2DNeuralNetwork(object):
    """
    A Conv2D Neural Network Model that can support an arbitrary numbers of
    layers.

    Attributes:
        filters: List of number of filters in each Conv2D layer
        kernel_sizes: List of kernel sizes in each Conv2D layer
        conv2d_activation: Type of activation function for conv2d layers
        pool_sizes: List of Max Pool sizes
        dense_sizes: Sizes of dense layers
        dense_activation: Type of activation function for dense layers
        output_activation: Type of activation function for output layer
        lr: Optimizer learning rate
        optimizer: Name of optimizer or optimizer object.
        adam_beta_1: Exponential decay rate for the first moment estimates
        adam_beta_2: Exponential decay rate for the first moment estimates
        sgd_momentum: Stochastic Gradient Descent momentum
        decay: Optimizer decay
        loss: Name of loss function or loss object
        batch_size: Number of examples per batch
        epochs: Number of epochs to train
        verbose: Level of detail to provide during training
        model: Keras Model object
    """
    def __init__(
        self, 
        filters=(8,), 
        kernel_sizes=(5,),
        conv2d_activation="relu", 
        pool_sizes=(4,), 
        pool_dropout=0.0,
        dense_sizes=(64,),
        dense_activation="relu", 
        dense_dropout = 0.0,
        output_activation="linear",
        lr=0.001, 
        optimizer="adam", 
        adam_beta_1=0.9,
        adam_beta_2=0.999, 
        sgd_momentum=0.9, 
        decay=0, 
        loss="mse",
        metrics = [], 
        batch_size=32, 
        epochs=2, 
        verbose=0
    ):
        
        self.filters = filters
        self.kernel_sizes = [tuple((v,v)) for v in kernel_sizes]
        self.conv2d_activation = conv2d_activation
        self.pool_sizes = [tuple((v,v)) for v in pool_sizes]
        self.pool_dropout = pool_dropout
        self.dense_sizes = dense_sizes
        self.dense_activation = dense_activation
        self.dense_dropout = dense_dropout
        self.output_activation = output_activation
        self.lr = lr
        self.optimizer = optimizer
        self.optimizer_obj = None
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.sgd_momentum = sgd_momentum
        self.decay = decay
        self.loss = loss
        self.metrics = metrics
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.model = None
        
        if self.conv2d_activation == "leakyrelu":
            self.conv2d_activation = LeakyReLU(alpha=0.1)
        if self.dense_activation == "leakyrelu":
            self.dense_activation = LeakyReLU(alpha=0.1)
        if self.output_activation == "leakyrelu":
            self.output_activation = LeakyReLU(alpha=0.1)

    def build_neural_network(self, input_shape, n_particles, output_shape):
        """Create Keras neural network model and compile it."""
        
        # Input
        conv_input = Input(shape=(input_shape), name="input")
        
        # ConvNet encoder
        nn_model = conv_input
        for h in range(len(self.filters)):
            nn_model = Conv2D(self.filters[h],
                              self.kernel_sizes[h],
                              padding="same",
                              activation=self.conv2d_activation,
                              kernel_initializer='he_uniform',
                              name=f"conv2D_{h:02d}")(nn_model)
            nn_model = MaxPool2D(self.pool_sizes[h], padding='same',
                                 name=f"maxpool2D_{h:02d}")(nn_model)
            if self.pool_dropout > 0.0:
                nn_model = Dropout(self.pool_dropout, 
                                   name = f"maxpool2D_dr_{h:02d}")(nn_model)
        nn_model = Flatten()(nn_model)
        
        # Classifier
        for h in range(len(self.dense_sizes)):
            nn_model = Dense(self.dense_sizes[h],
                             activation=self.dense_activation,
                             kernel_initializer='he_uniform',
                             name=f"dense_{h:02d}")(nn_model)
            if self.dense_dropout > 0.0:
                nn_model = Dropout(self.dense_dropout, 
                                   name=f"dense_dr_{h:02d}")(nn_model)
        
        # Output
        nn_model = RepeatVector(n_particles, name = "repeat")(nn_model)
        nn_model = Dense(output_shape,
                         activation=self.output_activation,
                         name=f"dense_output")(nn_model)
        nn_model = Lambda(
            self.LastLayer,
            input_shape = (n_particles, output_shape)
        )(nn_model)
        
        self.model = Model(conv_input, nn_model)
        
        if self.optimizer == "adam":
            self.optimizer_obj = Adam(lr=self.lr, clipnorm = 1.0)
        elif self.optimizer == "sgd":
            self.optimizer_obj = SGD(lr=self.lr, momentum=self.sgd_momentum,
                                     decay=self.decay)
            
        self.model.compile(
            optimizer=self.optimizer_obj, 
            loss=self.loss,
            metrics=self.metrics
        )
        #self.model.summary()

    def fit(self, x, y, xv=None, yv=None, callbacks=None):
        
        if len(x.shape[1:])==2:
            x = np.expand_dims(x, axis=-1)
        if len(y.shape) == 1:
            output_shape = 1
        else:
            output_shape = y.shape[1]
        
        input_shape = x.shape[1:]
        self.build_neural_network(input_shape, output_shape)
        self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs,
                       verbose=self.verbose, validation_data=(xv, yv), callbacks=callbacks)
        return self.model.history.history
    
    def LastLayer(self, x):
        return 1.75 * K.tanh(x / 100) 

    def predict(self, x):
        y_out = self.model.predict(np.expand_dims(x, axis=-1),
                                   batch_size=self.batch_size)
        return y_out

    def predict_proba(self, x):
        y_prob = self.model.predict(x, batch_size=self.batch_size)
        return y_prob
    
    def load_weights(self, weights):
        try:
            self.model.load_weights(weights)
            self.model.compile(
                optimizer=self.optimizer, 
                loss=self.loss, 
                metrics=self.metrics
            )
        except:
            print("You must first call build_neural_network before loading weights. Exiting.")
            sys.exit(1)