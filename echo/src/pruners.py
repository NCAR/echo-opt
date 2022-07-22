from optuna.pruners.__init__ import __all__ as supported_pruners
from tensorflow.python.keras.callbacks import Callback
from typing import Dict
import logging
import optuna
import warnings

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


def pruners(pruner):
    _type = pruner.pop("type")

    assert (
        _type in supported_pruners
    ), f"Pruner {_type} is not valid. Select from {supported_pruners}"

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


class KerasPruningCallback(Callback):
    def __init__(self, trial, monitor, interval=1):
        super(KerasPruningCallback, self).__init__()
        self.trial = trial
        self.monitor = monitor
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        # type: (int, Dict[str, float]) -> None
        logs = logs or {}
        current_score = logs.get(self.monitor)
        if current_score is None:
            return
        self.trial.report(current_score, step=epoch)
        if self.trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.structs.TrialPruned(message)
