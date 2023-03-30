import os
import random
import logging
import numpy as np
try:
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping
except ImportError as err:
    print("This example script requires tensorflow to be installed. Please install tensorflow before proceeding.")
    raise err

from echo.src.base_objective import BaseObjective
from optuna.integration import TFKerasPruningCallback
import optuna

import warnings

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


def seed_everything(seed=1234):
    """Set seeds for reproducibility"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def custom_updates(trial, conf):

    # Get list of hyperparameters from the config
    hyperparameters = conf["optuna"]["parameters"]

    # Now update some via custom rules
    filter1 = trial.suggest_int(**hyperparameters["filter1"]["settings"])
    filter2 = trial.suggest_int(**hyperparameters["filter2"]["settings"])
    learning_rate = trial.suggest_float(**hyperparameters["learning_rate"]["settings"])
    batch_size = trial.suggest_int(**hyperparameters["batch_size"]["settings"])
    dropout = trial.suggest_float(**hyperparameters["dropout"]["settings"])

    # Update the trial parameters in the configuration
    conf["filter1"] = filter1
    conf["filter2"] = filter2
    conf["learning_rate"] = learning_rate
    conf["batch_size"] = batch_size
    conf["dropout"] = dropout

    return conf


class Objective(BaseObjective):
    def __init__(self, config, metric="val_loss"):

        # Initialize the base class
        BaseObjective.__init__(self, config, metric)

    def train(self, trial, conf):

        # Custom updates
        # In this example, ECHO will automatically update
        # the model configuration. One would use such an example
        # when the model config cannot be structured in a way
        # so that ECHO will perform automatic updates.

        # conf = custom_updates(trial, conf)

        filter1 = conf["filter1"]
        filter2 = conf["filter2"]
        learning_rate = conf["learning_rate"]
        batch_size = conf["batch_size"]
        dropout = conf["dropout"]
        epochs = conf["epochs"]
        seed = conf["seed"]

        # Fix seed for reproducibility
        seed_everything(seed)

        # Load the CIFAR dataset
        num_classes = 10
        input_shape = (32, 32, 3)

        # Load the data and split it between train and test sets
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        logger.info(f"x_train/valid shape: {x_train.shape}")
        logger.info(f"train/valid samples: {x_train.shape[0]}")
        logger.info(f"test hold out samples: {x_test.shape[0]}")

        # convert class vectors to binary class matrices
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        # Load the model
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=input_shape),
                layers.Conv2D(filter1, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(filter2, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(dropout),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        # Load training callbacks
        callbacks = [
            EarlyStopping(**conf["callbacks"]["EarlyStopping"]),
            ReduceLROnPlateau(**conf["callbacks"]["ReduceLROnPlateau"]),
            TFKerasPruningCallback(trial, self.metric),
        ]

        # Compile
        model.compile(
            lr=learning_rate,
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

        # Train
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=0.2,
        )

        # Example of how to check if a trial could be pruned (maunally)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Return the validation accuracy for the last epoch.
        objective = max(history.history[self.metric])

        results_dictionary = {self.metric: objective}

        return results_dictionary
