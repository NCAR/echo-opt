import os
import yaml
import optuna
import logging
import warnings
from typing import Dict
from echo.src.pruners import pruners
from echo.src.samplers import samplers
try:
    # optuna >= 4.0: JournalFileBackend lives in optuna.storages.journal
    from optuna.storages.journal import JournalFileBackend
    from optuna.storages import JournalStorage
except ImportError:
    # optuna < 4.0: the class was called JournalFileStorage
    from optuna.storages import JournalStorage
    from optuna.storages import JournalFileStorage as JournalFileBackend


warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


def configure_storage(hyper_config):
    """Build and return an Optuna storage backend from the hyperparameter config.

    Parameters
    ----------
    hyper_config : dict
        Full hyperparameter config. Reads ``save_path``,
        ``optuna.storage_type``, and ``optuna.storage``.

    Returns
    -------
    str or JournalStorage
        - ``"sqlite"`` → ``sqlite:///{save_path}/{storage}`` URL string.
        - ``"maria"``  → the raw ``storage`` string (full MySQL URL).
        - ``"nfs"``    → a :class:`optuna.storages.JournalStorage` backed by a
          :class:`JournalFileBackend` at ``{save_path}/{storage}``. Use this
          for shared network filesystems where SQLite locking is unreliable.
    """
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
        storage = JournalStorage(JournalFileBackend(storage))
    return storage


def configure_sampler(hyper_config):
    """Instantiate and return the Optuna sampler specified in the config.

    If no ``sampler`` key is present under ``optuna``, defaults to
    :class:`optuna.samplers.TPESampler` for both single- and multi-objective
    studies.

    Parameters
    ----------
    hyper_config : dict
        Full hyperparameter config. Reads ``optuna.direction`` and
        ``optuna.sampler`` (optional).

    Returns
    -------
    optuna.samplers.BaseSampler
        Configured sampler instance.
    """
    direction = hyper_config["optuna"]["direction"]
    single_objective = isinstance(direction, str)
    if "sampler" not in hyper_config["optuna"]:
        logger.warning("No sampler was supplied in the hyperparameter config file.")
        if single_objective:  # single-objective
            logger.warning("\tUsing the default TPESampler class.")
            sampler = optuna.samplers.TPESampler()
        else:  # multi-objective: optuna 3+ uses TPESampler natively
            logger.warning("\tUsing the default TPESampler class (multi-objective).")
            sampler = optuna.samplers.TPESampler()
    else:
        sampler = samplers(hyper_config["optuna"]["sampler"])
    return sampler


def configure_pruner(hyper_config):
    """Instantiate and return the Optuna pruner specified in the config.

    If no ``pruner`` key is present under ``optuna``, defaults to
    :class:`optuna.pruners.NopPruner` (no pruning).

    Parameters
    ----------
    hyper_config : dict
        Full hyperparameter config. Reads ``optuna.pruner`` (optional).

    Returns
    -------
    optuna.pruners.BasePruner
        Configured pruner instance.
    """
    if "pruner" not in hyper_config["optuna"]:
        logger.warning("No pruner was supplied in the hyperparameter config file.")
        logger.warning("\tUsing the default NopPruner class (no pruning).")
        pruner = optuna.pruners.NopPruner()
    else:
        pruner = pruners(hyper_config["optuna"]["pruner"])
    return pruner


def recursive_config_reader(_dict: Dict[str, str], path: bool = None):
    """Yield ``(key_path, value)`` for every leaf node in a nested dict.

    Parameters
    ----------
    _dict : dict
        Arbitrarily nested dictionary.
    path : list, optional
        Accumulated key path (used internally for recursion).

    Yields
    ------
    tuple[list[str], any]
        ``(key_path, leaf_value)`` where ``key_path`` is a list of keys
        leading to the leaf from the top-level dict.
    """

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
    """Update a nested dictionary at the location specified by ``nested_keys``.

    Parameters
    ----------
    nested_keys : list[str]
        Ordered list of keys forming the path to the target leaf, e.g.
        ``["optimizer", "learning_rate"]``.
    dictionary : dict
        The dict to update in-place.
    update : any
        The new value to assign at the leaf.
    """
    if isinstance(dictionary, dict) and len(nested_keys) > 1:
        recursive_update(nested_keys[1:], dictionary[nested_keys[0]], update)
    else:
        dictionary[nested_keys[0]] = update


def config_check(hyper_config, model_config, file_check=False):
    """Validate the hyperparameter and model configuration dictionaries.

    Raises ``AssertionError`` for any missing or invalid field. Can optionally
    load configs from file paths instead of pre-parsed dicts.

    Parameters
    ----------
    hyper_config : dict or str
        Hyperparameter config dict, or a file path if ``file_check=True``.
    model_config : dict or str
        Model config dict, or a file path if ``file_check=True``.
    file_check : bool, optional
        If True, treat both arguments as file paths and load them. Also
        verifies that the files exist. Default is False.

    Returns
    -------
    bool
        Always returns True on success; raises on failure.
    """

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
