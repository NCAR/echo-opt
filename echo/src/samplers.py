import optuna
from optuna.samplers import (
    TPESampler,
    RandomSampler,
    GridSampler,
    CmaEsSampler,
    NSGAIISampler,
    PartialFixedSampler,
    BaseSampler,
)
try:
    # optuna >= 4.0 moved these to optuna.search_space
    from optuna.search_space import IntersectionSearchSpace, intersection_search_space
except ImportError:
    from optuna.samplers import IntersectionSearchSpace, intersection_search_space
import logging
import warnings

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)

SUPPORTED_SAMPLERS = [
    "TPESampler",
    "GridSampler",
    "RandomSampler",
    "CmaEsSampler",
    "NSGAIISampler",
    "PartialFixedSampler",
    "BaseSampler",
    "IntersectionSearchSpace",
    "intersection_search_space",
]


def samplers(sampler):
    _type = sampler.pop("type")

    assert (
        _type in SUPPORTED_SAMPLERS
    ), f"Sampler {_type} is not valid. Select from {SUPPORTED_SAMPLERS}"

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
    if _type == "BaseSampler":
        return BaseSampler(**sampler)
    if _type == "NSGAIISampler":
        return NSGAIISampler(**sampler)
    if _type == "PartialFixedSampler":
        return PartialFixedSampler(**sampler)
    if _type == "intersection_search_space":
        return intersection_search_space(**sampler)
