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
        # suggest_discrete_uniform was removed in optuna 4.0; map to suggest_float with step
        settings = dict(config["settings"])
        q = settings.pop("q", None)
        if q is not None:
            settings["step"] = q
        return float(trial.suggest_float(**settings))
    if _type == "float":
        return float(trial.suggest_float(**config["settings"]))
    if _type == "int":
        return int(trial.suggest_int(**config["settings"]))
    if _type == "loguniform":
        # suggest_loguniform was removed in optuna 4.0; map to suggest_float with log=True
        settings = dict(config["settings"])
        settings["log"] = True
        return float(trial.suggest_float(**settings))
    if _type == "uniform":
        # suggest_uniform was removed in optuna 4.0; map to suggest_float
        return float(trial.suggest_float(**config["settings"]))
