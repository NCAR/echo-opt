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
    """Call the appropriate ``trial.suggest_*`` method for a parameter config.

    Supports both current Optuna 4.x types and legacy aliases for backward
    compatibility with older YAML configs.

    Parameters
    ----------
    trial : optuna.Trial
        The active trial object.
    config : dict
        Parameter config with keys:

        - ``"type"`` : one of ``supported_trials``
        - ``"settings"`` : dict of keyword arguments forwarded to the suggest
          method (e.g. ``name``, ``low``, ``high``, ``choices``, ``step``).

        Legacy types are remapped internally:

        ==================== ==========================================
        YAML type            Optuna call
        ==================== ==========================================
        ``"loguniform"``     ``suggest_float(..., log=True)``
        ``"uniform"``        ``suggest_float(...)``
        ``"discrete_uniform"`` ``suggest_float(..., step=q)``
        ==================== ==========================================

    Returns
    -------
    float, int, or str
        The value suggested by Optuna for this trial.

    Raises
    ------
    AssertionError
        If ``config["type"]`` is not in ``supported_trials``.
    """

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
