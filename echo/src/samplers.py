from optuna.samplers._tpe.sampler import TPESampler
from optuna.samplers._tpe.multi_objective_sampler import MOTPESampler
from optuna.samplers._search_space import IntersectionSearchSpace
from optuna.samplers._search_space import intersection_search_space
from optuna.samplers._random import RandomSampler
from optuna.samplers._partial_fixed import PartialFixedSampler
from optuna.samplers import NSGAIISampler
from optuna.samplers._grid import GridSampler
from optuna.samplers._cmaes import CmaEsSampler
from optuna.samplers._base import BaseSampler
from optuna.samplers.__init__ import __all__ as supported_samplers
import logging
import warnings

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


def samplers(sampler):
    _type = sampler.pop("type")

    assert (
        _type in supported_samplers
    ), f"Sampler {_type} is not valid. Select from {supported_samplers}"

    if _type == "TPESampler":
        return TPESampler(**sampler)
    if _type == "GridSampler":
        if "search_space" not in sampler:
            raise OSError("You must provide search_space options with the GridSampler.")
        else:
            return GridSampler(**sampler)
    if _type == "RandomSampler":
        return RandomSampler(**sampler)
    if _type == "CmaEsSampler":
        return CmaEsSampler(**sampler)
    if _type == "IntersectionSearchSpace":
        return IntersectionSearchSpace(**sampler)
    if _type == "MOTPESampler":
        return MOTPESampler(**sampler)
    if _type == "BaseSampler":
        return BaseSampler(**sampler)
    if _type == "NSGAIISampler":
        return NSGAIISampler(**sampler)
    if _type == "PartialFixedSampler":
        return PartialFixedSampler(**sampler)
    if _type == "intersection_search_space":
        return intersection_search_space(**sampler)
