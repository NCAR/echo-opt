from tensorflow.keras.callbacks import Callback
from typing import Dict
import logging
import optuna
import warnings
warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)

supported_pruners = [
    "MedianPruner"
]


def pruners(pruner):
    _type = pruner.pop("type")
    if _type not in supported_pruners:
        message = f"Pruner {_type} is not valid. Select from {supported_pruners}"
        logger.warning(message)
        raise OSError(message)
    if _type == "Median":
        return optuna.pruners.MedianPruner(**pruner)


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
