import logging
import warnings
import subprocess
import numpy as np
import pandas as pd
from collections import Counter

import collections
from typing import Any
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple
from optuna.trial._state import TrialState
from optuna import multi_objective
import optuna

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


def gpu_report():

    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    cmd = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"]
    result = subprocess.check_output(cmd)
    result = result.decode("utf-8")
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split("\n")]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def devices(gpu=False):
    if bool(gpu):
        try:
            _gpu_report = sorted(gpu_report().items(), key=lambda x: x[1], reverse=True)
            if len(_gpu_report) > 1:
                device = [x[0] for x in _gpu_report]
            else:
                device = _gpu_report[0][0]
        except Exception as E:
            logger.warning(
                f"The gpu is not responding to a call from nvidia-smi.\
                Setting device = cpu, but this may fail. {str(E)}"
            )
            device = 0
    else:
        device = "cpu"
    logger.info(f"Using device {device}")
    return device


def get_sec(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def successful_trials(study):
    total_completed_trials = 0
    for trial in study.get_trials():
        if str(trial.state.name) in ["COMPLETE", "PRUNED"]:
            total_completed_trials += 1
    return total_completed_trials


def trial_report(study):
    states = [t.state.name for t in study.get_trials()]
    state_histo = Counter(states)
    return state_histo


def to_df(study):
    attrs = (
        "number",
        "values",
        # "intermediate_values",
        "datetime_start",
        "datetime_complete",
        "params",
        "user_attrs",
        "system_attrs",
        "state",
    )
    multi_index = False

    trials = study.get_trials(deepcopy=False)

    attrs_to_df_columns = collections.OrderedDict()
    for attr in attrs:
        if attr.startswith("_"):
            # Python conventional underscores are omitted in the dataframe.
            df_column = attr[1:]
        else:
            df_column = attr
        attrs_to_df_columns[attr] = df_column

    # column_agg is an aggregator of column names.
    # Keys of column agg are attributes of `FrozenTrial` such as 'trial_id' and 'params'.
    # Values are dataframe columns such as ('trial_id', '') and ('params', 'n_layers').
    column_agg: DefaultDict[str, Set] = collections.defaultdict(set)
    non_nested_attr = ""

    def _create_record_and_aggregate_column(
        trial: "optuna.trial.FrozenTrial",
    ) -> Dict[Tuple[str, str], Any]:

        n_objectives = len(study.directions)
        trial = multi_objective.trial.FrozenMultiObjectiveTrial(
            n_objectives,
            trial,
        )._trial

        record = {}
        for attr, df_column in attrs_to_df_columns.items():
            value = getattr(trial, attr)
            if isinstance(value, TrialState):
                # Convert TrialState to str and remove the common prefix.
                value = str(value).split(".")[-1]
            if isinstance(value, dict):
                for nested_attr, nested_value in value.items():
                    record[(df_column, nested_attr)] = nested_value
                    column_agg[attr].add((df_column, nested_attr))
            elif isinstance(value, list):
                # Expand trial.values.
                for nested_attr, nested_value in enumerate(value):
                    record[(df_column, nested_attr)] = nested_value
                    column_agg[attr].add((df_column, nested_attr))
            elif attr == "values":
                # trial.values should be None when the trial's state is FAIL or PRUNED.
                # assert value is None
                if value is None:
                    value = [None for k in range(study.n_objectives)]

                for nested_attr, nested_value in enumerate(value):
                    record[(df_column, nested_attr)] = nested_value
                    column_agg[attr].add((df_column, nested_attr))
            else:
                record[(df_column, non_nested_attr)] = value
                column_agg[attr].add((df_column, non_nested_attr))
        return record

    records = [_create_record_and_aggregate_column(trial) for trial in trials]

    columns: List[Tuple[str, str]] = sum(
        (sorted(column_agg[k]) for k in attrs if k in column_agg), []
    )

    df = pd.DataFrame(records, columns=pd.MultiIndex.from_tuples(columns))

    if not multi_index:
        # Flatten the `MultiIndex` columns where names are concatenated with underscores.
        # Filtering is required to omit non-nested columns avoiding unwanted trailing
        # underscores.
        df.columns = [
            "_".join(filter(lambda c: c, map(lambda c: str(c), col))) for col in columns
        ]
    return df


def study_report(study, hyper_config):
    n_trials = hyper_config["optuna"]["n_trials"]
    state_histo = trial_report(study)
    logger.info("Summary statistics for the current study:")
    logger.info(f"\tTotal number of trials in the study: {len(study.get_trials())}")
    for key, val in state_histo.items():
        key = str(key) if "." not in str(key) else str(key).split(".")[-1]
        logger.info(f"\tTrials with state {key}: {val}")
    logger.info(f"\tRequested number of trials: {n_trials}")
    total_completed_trials = successful_trials(study)

    if total_completed_trials > 1:
        logger.info("\t...")
        if not isinstance(hyper_config["optuna"]["metric"], list):
            df = study.trials_dataframe()
        else:
            df = study.trials_dataframe()
            # df = to_df(study)
        df["run_time"] = df["datetime_complete"] - df["datetime_start"]
        completed_runs = df["datetime_complete"].apply(lambda x: True if x else False)
        run_time = df["run_time"][completed_runs].apply(
            lambda x: x.total_seconds() / 3600.0
        )
        logger.info(f"\tTotal study simulation run time: {run_time.sum():.4f} hrs")
        logger.info(f"\tAverage trial simulation run time: {run_time.mean():.4f} hrs")
        logger.info(f"\tThe longest trial took {run_time.max():.4f} hrs")

    if (total_completed_trials < n_trials) and (total_completed_trials > 1):

        trails_remaining = n_trials - total_completed_trials
        time_needed = trails_remaining * run_time.mean()
        logger.info(
            f"\tEstimated remaining simulation time needed: {time_needed:.4f} hrs"
        )

        if "pbs" in hyper_config:
            for option in hyper_config["pbs"]["batch"]["l"]:
                if "walltime" in option:
                    time_str = option.split("=")[1]
                    if ":" in time_str:
                        walltime = get_sec(option.split("=")[1])
                    else:
                        walltime = False
            if walltime:
                nodes = int(np.ceil(3600 * time_needed / walltime))
                logger.info(
                    f"\tWith a given wall-time of {time_str}, submit {nodes} PBS workers to complete the study"
                )

        if "slurm" in hyper_config:
            time_str = hyper_config["slurm"]["batch"]["t"]
            if ":" in time_str:
                walltime = get_sec(time_str)
            else:
                walltime = False
            if walltime:
                nodes = int(np.ceil(3600 * time_needed / walltime))
                logger.info(
                    f"\tWith a given wall-time of {time_str}, submit {nodes} SLURM workers to complete the study"
                )

    return total_completed_trials
