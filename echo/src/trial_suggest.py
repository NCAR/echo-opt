import logging
import warnings

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


supported_trials = [
    "categorical",
    "discrete_uniform",
    "float",
    "int",
    "loguniform",
    "uniform",
]


def trial_suggest_loader(trial, config):

    _type = config["type"]

    assert (
        _type in supported_trials
    ), f"Type {_type} is not valid. Select from {supported_trials}"

    if _type == "categorical":
        return trial.suggest_categorical(**config["settings"])
    if _type == "discrete_uniform":
        return int(trial.suggest_discrete_uniform(**config["settings"]))
    if _type == "float":
        return float(trial.suggest_float(**config["settings"]))
    if _type == "int":
        return int(trial.suggest_int(**config["settings"]))
    if _type == "loguniform":
        return float(trial.suggest_loguniform(**config["settings"]))
    if _type == "uniform":
        return float(trial.suggest_uniform(**config["settings"]))
