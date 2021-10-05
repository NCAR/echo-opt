import warnings
warnings.filterwarnings("ignore")

import sys
import optuna
import logging


logger = logging.getLogger(__name__)


supported_trials = [
    "categorical",
    "discrete_uniform",
    "float",
    "int",
    "loguniform",
    "uniform"
]


def trial_suggest_loader(trial, config):
    
    try:
        _type = config["type"]
        if _type == "categorical":
            return trial.suggest_categorical(**config["settings"])
        elif _type == "discrete_uniform":
            return int(trial.suggest_discrete_uniform(**config["settings"]))
        elif _type == "float":
            return float(trial.suggest_float(**config["settings"]))
        elif _type == "int":
            return int(trial.suggest_int(**config["settings"]))
        elif _type == "loguniform":
            return float(trial.suggest_loguniform(**config["settings"]))
        elif _type == "uniform":
            return float(trial.suggest_uniform(**config["settings"]))
        else: #if _type not in supported_trials:
            message = f"Type {_type} is not valid. Select from {supported_trials}"
            logger.warning(message)
            raise OSError(message)
    except Exception as E:
        print("FAILED IN TRIAL SUGGEST", E, config)
        raise OSError