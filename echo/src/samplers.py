import warnings
warnings.filterwarnings("ignore")

import sys
import optuna
import logging


logger = logging.getLogger(__name__)


supported_samplers = [
    "TPESampler",
    "GridSampler",
    "RandomSampler",
    "CmaEsSampler",
    "IntersectionSearchSpace",
    "MOTPEMultiObjectiveSampler",
    "NSGAIIMultiObjectiveSampler",
    "RandomMultiObjectiveSampler"
]


def samplers(sampler):
    _type = sampler.pop("type")
    if _type not in supported_samplers:
        message = f"Sampler {_type} is not valid. Select from {supported_samplers}"
        logger.warning(message)
        raise OSError(message)
    if _type == "TPESampler":
        return optuna.samplers.TPESampler(**sampler)
    elif _type == "GridSampler":
        if "search_space" not in sampler:
            raise OSError("You must provide search_space options with the GridSampler.")
        else:
            return optuna.samplers.GridSampler(**sampler)
    elif _type == "RandomSampler":
        return optuna.samplers.RandomSampler(**sampler)
    elif _type == "CmaEsSampler":
        return optuna.integration.CmaEsSampler(**sampler)
    elif _type == "IntersectionSearchSpace":
        return optuna.integration.IntersectionSearchSpace(**sampler)
    # support for multi-objective studies
    elif _type == "MOTPEMultiObjectiveSampler":
        return optuna.multi_objective.samplers.MOTPEMultiObjectiveSampler(**sampler)
    elif _type == "NSGAIIMultiObjectiveSampler":
        return optuna.multi_objective.samplers.NSGAIIMultiObjectiveSampler(**sampler)
    elif _type == "RandomMultiObjectiveSampler":
        return optuna.multi_objective.samplers.RandomMultiObjectiveSampler(**sampler)