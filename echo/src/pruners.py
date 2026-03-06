from typing import Dict
import logging
import optuna
import warnings

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)

SUPPORTED_PRUNERS = [
    "BasePruner",
    "HyperbandPruner",
    "MedianPruner",
    "NopPruner",
    "PatientPruner",
    "PercentilePruner",
    "SuccessiveHalvingPruner",
    "ThresholdPruner",
]


def pruners(pruner):
    """Instantiate an Optuna pruner from a config dict.

    Parameters
    ----------
    pruner : dict
        Must contain a ``"type"`` key matching one of ``SUPPORTED_PRUNERS``.
        All remaining keys are forwarded as keyword arguments to the pruner
        constructor. **Note:** the ``"type"`` key is popped from the dict
        in-place.

    Returns
    -------
    optuna.pruners.BasePruner
        Configured pruner instance.

    Raises
    ------
    AssertionError
        If ``pruner["type"]`` is not in ``SUPPORTED_PRUNERS``.
    """
    pruner = dict(pruner)  # copy so the caller's dict is not mutated
    _type = pruner.pop("type")

    assert (
        _type in SUPPORTED_PRUNERS
    ), f"Pruner {_type} is not valid. Select from {SUPPORTED_PRUNERS}"

    if _type == "BasePruner":
        return optuna.pruners.BasePruner(**pruner)
    if _type == "HyperbandPruner":
        return optuna.pruners.HyperbandPruner(**pruner)
    if _type == "MedianPruner":
        return optuna.pruners.MedianPruner(**pruner)
    if _type == "NopPruner":
        return optuna.pruners.NopPruner(**pruner)
    if _type == "PatientPruner":
        return optuna.pruners.PatientPruner(**pruner)
    if _type == "PercentilePruner":
        return optuna.pruners.PercentilePruner(**pruner)
    if _type == "SuccessiveHalvingPruner":
        return optuna.pruners.SuccessiveHalvingPruner(**pruner)
    if _type == "ThresholdPruner":
        return optuna.pruners.ThresholdPruner(**pruner)


class KerasPruningCallback(object):
    """Keras-compatible callback that integrates Optuna trial pruning.

    On each epoch end, the monitored metric is reported to the trial. If the
    trial's pruner decides the trial should be stopped, :class:`optuna.TrialPruned`
    is raised, which Optuna treats as a pruned (but not failed) trial.

    Parameters
    ----------
    trial : optuna.Trial
        The current Optuna trial.
    monitor : str
        The metric key to read from the Keras ``logs`` dict (e.g.
        ``"val_loss"``).
    interval : int, optional
        Epoch interval at which to report; currently unused (every epoch is
        always reported). Default is 1.

    Examples
    --------
    .. code-block:: python

        callbacks = [KerasPruningCallback(trial, monitor="val_loss")]
        model.fit(..., callbacks=callbacks)
    """
    def __init__(self, trial, monitor, interval=1):
        self.trial = trial
        self.monitor = monitor
        self.interval = interval
        self.validation_data = None
        self.model = None
        # Whether this Callback should only run on the chief worker in a
        # Multi-Worker setting.
        self._chief_worker_only = None
        self._supports_tf_logs = False
        self.params = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        # type: (int, Dict[str, float]) -> None
        logs = logs or {}
        current_score = logs.get(self.monitor)
        if current_score is None:
            return
        self.trial.report(current_score, step=epoch)
        if self.trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)
