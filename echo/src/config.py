import os
import yaml
import optuna
import logging
import warnings
from typing import Dict
from echo.src.pruners import pruners
from echo.src.samplers import samplers
from optuna.storages import JournalStorage, JournalFileStorage


warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


def configure_storage(hyper_config):
    # Set up storage db
    save_path = hyper_config["save_path"]
    storage_type = hyper_config["optuna"]["storage_type"]
    storage = hyper_config["optuna"]["storage"]
    if storage_type == "sqlite":
        storage = os.path.join(save_path, storage)
        storage = f"sqlite:///{storage}"
    elif storage_type == "maria":
        storage = storage
    elif storage_type == "nfs":
        storage = os.path.join(save_path, storage)
        storage = JournalStorage(JournalFileStorage(storage))
    return storage


def configure_sampler(hyper_config):
    direction = hyper_config["optuna"]["direction"]
    single_objective = isinstance(direction, str)
    if "sampler" not in hyper_config["optuna"]:
        logger.warning("No sampler was supplied in the hyperparameter config file.")
        if single_objective:  # single-objective
            logger.warning("\tUsing the default TPESampler class.")
            sampler = optuna.samplers.TPESampler()
        else:  # multi-objective equivalent of TPESampler
            logger.warning("\tUsing the default MOTPEMultiObjectiveSampler class.")
            sampler = optuna.multi_objective.samplers.MOTPEMultiObjectiveSampler()
    else:
        sampler = samplers(hyper_config["optuna"]["sampler"])
    return sampler


def configure_pruner(hyper_config):
    if "pruner" not in hyper_config["optuna"]:
        logger.warning("No pruner was supplied in the hyperparameter config file.")
        logger.warning("\tUsing the default NopPruner class (no pruning).")
        pruner = optuna.pruners.NopPruner()
    else:
        pruner = pruners(hyper_config["optuna"]["pruner"])
    return pruner


def recursive_config_reader(_dict: Dict[str, str], path: bool = None):

    if path is None:
        path = []
    for k, v in _dict.items():
        newpath = path + [k]
        if isinstance(v, dict):
            for u in recursive_config_reader(v, newpath):
                yield u
        else:
            yield newpath, v


def recursive_update(nested_keys, dictionary, update):
    if isinstance(dictionary, dict) and len(nested_keys) > 1:
        recursive_update(nested_keys[1:], dictionary[nested_keys[0]], update)
    else:
        dictionary[nested_keys[0]] = update


def config_check(hyper_config, model_config, file_check=False):

    if file_check:
        assert os.path.isfile(
            hyper_config
        ), f"Hyperparameter optimization config file {hyper_config} does not exist"
        with open(hyper_config) as f:
            hyper_config = yaml.load(f, Loader=yaml.FullLoader)

        """ Check if model config file exists """
        assert os.path.isfile(
            model_config
        ), f"Model config file {model_config} does not exist"
        with open(model_config) as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)

    """ Save path must be defined """
    assert (
        "save_path" in hyper_config
    ), "You must specify the save_path in the hyperparameter config"

    """ Check if the wall-time exists """
    if "slurm" in hyper_config:
        assert (
            "t" in hyper_config["slurm"]["batch"]
        ), "You must supply a wall time in the hyperparameter config at slurm:batch:t"

    if "pbs" in hyper_config:
        assert any(
            [("walltime" in x) for x in hyper_config["pbs"]["batch"]["l"]]
        ), "You must supply a wall time in the hyperparameter config at pbs:bash:l"

    """ Check if path to objective method exists """
    assert os.path.isfile(
        hyper_config["optuna"]["objective"]
    ), f'The objective file {hyper_config["optuna"]["objective"]} does not exist'

    """ Check if the optimization metric direction is supported """
    direction = hyper_config["optuna"]["direction"]
    single_objective = isinstance(direction, str)

    if single_objective:
        assert direction in [
            "maximize",
            "minimize",
        ], f"Optimizer direction {direction} not recognized. Choose from maximize or minimize"
    else:
        for direc in direction:
            assert direc in [
                "maximize",
                "minimize",
            ], f"Optimizer direction {direc} not recognized. Choose from maximize or minimize"

    """ Check the storage requirements """
    storage_type = hyper_config["optuna"]["storage_type"]
    assert storage_type in [
        "sqlite",
        "maria",
        "nfs",
    ], f"The storage type {storage_type} is not supported. Select from sqlite, maria, or nfs"

    return True
