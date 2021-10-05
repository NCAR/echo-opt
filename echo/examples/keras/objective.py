import warnings
warnings.filterwarnings("ignore")

import copy
import optuna
import logging
import traceback

from aimlutils.hyper_opt.base_objective import *
from data_generator import DataGenerator
from model import Conv2DNeuralNetwork

from holodecml.callbacks import get_callbacks
from aimlutils.hyper_opt.utils import KerasPruningCallback


logger = logging.getLogger(__name__)


def custom_updates(trial, conf):
    
    # Get list of hyperparameters from the config
    hyperparameters = conf["optuna"]["parameters"]
    
    # Now update some via custom rules
    filter1 = trial.suggest_int(**hyperparameters["filter1"]["settings"])
    filter2 = trial.suggest_int(**hyperparameters["filter2"]["settings"])
    filter3 = trial.suggest_int(**hyperparameters["filter3"]["settings"])
    kernel1 = trial.suggest_int(**hyperparameters["kernel1"]["settings"])
    kernel2 = trial.suggest_int(**hyperparameters["kernel2"]["settings"])
    kernel3 = trial.suggest_int(**hyperparameters["kernel3"]["settings"])
    pool1 = trial.suggest_int(**hyperparameters["pool1"]["settings"])
    pool2 = trial.suggest_int(**hyperparameters["pool2"]["settings"])
    pool3 = trial.suggest_int(**hyperparameters["pool3"]["settings"])
    dense1 = trial.suggest_int(**hyperparameters["dense1"]["settings"])
    dense2 = trial.suggest_int(**hyperparameters["dense2"]["settings"])
    
    conf["conv2d_network"]["filters"] = [filter1, filter2, filter3]
    conf["conv2d_network"]["kernel_sizes"] = [kernel1, kernel2, kernel3]
    conf["conv2d_network"]["pool_sizes"] = [pool1, pool2, pool3]
    conf["conv2d_network"]["dense_sizes"] = [dense1, dense2]
    
    return conf


class Objective(BaseObjective):
    
    def __init__(self, study, config, metric = "val_loss", device = "cpu"):
        
        # Initialize the base class
        BaseObjective.__init__(self, study, config, metric, device)


    def train(self, trial, conf):   
        
        # Custom updates
        conf = custom_updates(trial, conf)
        
        # Set up some globals
        path_data = conf["path_data"]
        num_particles = conf["num_particles"]
        split = 'train'
        subset = False
        output_cols = ["x", "y", "z", "d", "hid"]

        input_shape = (600, 400, 1)
        batch_size = conf["conv2d_network"]["batch_size"]
        n_particles = conf["num_particles"]
        output_channels = len(output_cols) - 1
        
        # Load the data
        train_gen = DataGenerator(
            path_data, num_particles, "train", subset, 
            output_cols, batch_size, maxnum_particles = 3, shuffle = False
        )
        train_scalers = train_gen.get_transform()
        valid_gen = DataGenerator(
            path_data, num_particles, "test", subset, 
            output_cols, batch_size, scaler = train_scalers, maxnum_particles = 3, shuffle = False
        )
        
        # Load the model
        model = Conv2DNeuralNetwork(**conf["conv2d_network"])    
        model.build_neural_network(input_shape, n_particles, output_channels)

        # Load callbacks
        callbacks = get_callbacks(conf["callbacks"])
        
        # Load optuna keras pruning callback
        pruning_callback = KerasPruningCallback(trial, self.metric)
        callbacks.append(pruning_callback)
        
        # Train a model
        try: # Aim to catch instances when the GPU memory overflows
            blackbox = model.model.fit(
                train_gen,
                validation_data=valid_gen,
                epochs=conf["conv2d_network"]["epochs"],
                verbose=True,
                callbacks=callbacks,
                use_multiprocessing=True,
                workers=8,
                max_queue_size=100
            )
        except: # When that happens, let optuna consider it as a pruned trial
            raise optuna.TrialPruned()
        
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Return the validation accuracy for the last epoch.
        objective = blackbox.history[self.metric][-1]
        
        results_dictionary = {
            self.metric: objective
        }
        
        return results_dictionary