import os
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

import tensorflow as tf


num_particles_dict = {
    1: '1particle',
    3: '3particle',
    'multi': 'multiparticle',
    '50-100': '50-100'}

split_dict = {
    'train' : 'training',
    'test'   : 'test',
    'valid': 'validation'}


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(
        
        self, 
        path_data: str, 
        num_particles: int, 
        split: str, 
        subset: bool, 
        output_cols: List[str], 
        batch_size: int, 
        shuffle: bool = True,
        maxnum_particles: int = False,
        scaler: Dict[str, str] = False) -> None:
        
        'Initialization'
        self.ds = self.open_dataset(path_data, num_particles, split)
        self.batch_size = batch_size
        self.output_cols = [x for x in output_cols if x != 'hid']        
        self.subset = subset
        self.hologram_numbers = self.ds.hologram_number.values
        if shuffle:
            random.shuffle(self.hologram_numbers)
        self.num_particles = num_particles
        self.xsize = len(self.ds.xsize.values)
        self.ysize = len(self.ds.ysize.values)
        self.shuffle = shuffle
        self.maxnum_particles = maxnum_particles
                
        if not scaler:
            self.scaler = {col: StandardScaler() for col in output_cols}
            for col in output_cols:
                scale = self.ds[col].values
                self.scaler[col].fit(scale.reshape(scale.shape[-1], -1))
        else:
            self.scaler = scaler
        
    def get_transform(self):
        return self.scaler

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(len(self.hologram_numbers) / self.batch_size)
    
    def __getitem__(self, idx):
        'Generate one batch of data'
        holograms = self.hologram_numbers[
            idx * self.batch_size: (idx + 1) * self.batch_size
        ]
        x_out, y_out, w_out = self._batch(holograms)
        return x_out, y_out, w_out
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            random.shuffle(self.hologram_numbers)
            
    def _batch(self, holograms: List[int]):
        'Create a batch of data'
        try:
        
            x_out = np.zeros((
                len(holograms), self.xsize, self.ysize
            ))
            y_out = np.zeros((
                len(holograms), 
                self.maxnum_particles if self.maxnum_particles else self.num_particles, 
                len(self.output_cols)
            ))
            # Move the scaler.transform to here
            
            a = time.time()
            for k, hologram in enumerate(holograms):
                im = self.ds["image"][hologram].values
                x_out[k] = (im-np.mean(im)) / (np.std(im))
                #A = np.log(np.abs(OpticsFFT(A)))                    
                particles = np.where(self.ds["hid"] == hologram + 1)[0]  
                for l, p in enumerate(particles):
                    for m, col in enumerate(self.output_cols):
                        val = self.ds[col][p].values
                        y_out[k, l, m] = self.scaler[col].transform(
                            val.reshape(1, -1)
                        )
                if self.maxnum_particles and len(particles) < self.maxnum_particles:
                    for l in range(len(particles), self.maxnum_particles):
                        for m, col in enumerate(self.output_cols):
                            val = y_out[k, l, m]
                            y_out[k, l, m] = self.scaler[col].transform(
                                val.reshape(1, -1)
                            )
            #
            # convert y_out to sparse if we are using padding
#             if self.maxnum_particles:
#                 y_out = sparse_vstack([
#                     csr_matrix(y_out[i]) for i in y_out.shape[0]
#                 ])
            
            x_out = np.expand_dims(x_out, axis=-1)
            return x_out, y_out, [None] #class weights option
        
        except:
            print(traceback.print_exc())
    
    def open_dataset(self, path_data, num_particles, split):
        """
        Opens a HOLODEC file

        Args: 
            path_data: (str) Path to dataset directory
            num_particles: (int or str) Number of particles per hologram
            split: (str) Dataset split of either 'train', 'valid', or 'test'

        Returns:
            ds: (xarray Dataset) Opened dataset
        """
        path_data = os.path.join(path_data, self.dataset_name(num_particles, split))

        if not os.path.isfile(path_data):
            print(f"Data file does not exist at {path_data}. Exiting.")
            raise 

        ds = xr.open_dataset(path_data)
        return ds
    
    def dataset_name(self, num_particles, split, file_extension='nc'):
        """
        Return the dataset filename given user inputs

        Args: 
            num_particles: (int or str) Number of particles per hologram
            split: (str) Dataset split of either 'train', 'valid', or 'test'
            file_extension: (str) Dataset file extension

        Returns:
            ds_name: (str) Dataset name
        """

        valid = [1,3,'multi','50-100']
        if num_particles not in valid:
            raise ValueError("results: num_particles must be one of %r." % valid)
        num_particles = num_particles_dict[num_particles]

        valid = ['train','test','valid']
        if split not in valid:
            raise ValueError("results: split must be one of %r." % valid)
        split = split_dict[split]
        ds_name = f'synthetic_holograms_{num_particles}_{split}.{file_extension}'

        return ds_name