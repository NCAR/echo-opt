import os
import optuna
import logging
import warnings
import subprocess
import numpy as np
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
    cmd = ['nvidia-smi', '--query-gpu=memory.free',
           '--format=csv,nounits,noheader']
    result = subprocess.check_output(cmd)
    result = result.decode('utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def devices(gpu = False):
    if bool(gpu):
        try:
            _gpu_report = sorted(
                gpu_report().items(),
                key=lambda x: x[1],
                reverse=True
            )
            if len(_gpu_report) > 1:
                device = [x[0] for x in _gpu_report]
            else:
                device = _gpu_report[0][0]
        except Exception as E:
            logger.warning(
                f"The gpu is not responding to a call from nvidia-smi.\
                Setting gpu device = 0, but this may fail. {str(E)}"
            )
            device = 0
    else:
        device = 'cpu'
    logger.info(f"Using device {device}")
    return device


def get_sec(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


def successful_trials(study):
    total_completed_trials = 0
    for trial in study.get_trials():
        if str(trial.state) in ["TrialState.COMPLETE", "TrialState.PRUNED"]:
            total_completed_trials += 1
    return total_completed_trials


def study_report(study, hyper_config):
    n_trials = hyper_config["optuna"]["n_trials"]
    total_completed_trials = successful_trials(study)
    logger.info("Summary statistics for the current study:")
    logger.info(f"\tTotal number of trials in the study: {len(study.get_trials())}")
    logger.info(f"\tCompleted / pruned trials: {total_completed_trials}")
    logger.info(f"\tRequested number of trials: {n_trials}")
    
    if total_completed_trials > 1:
        logger.info(f"\t...")
        df = study.trials_dataframe()
        df["run_time"] = df["datetime_complete"] - df["datetime_start"]
        completed_runs = df["datetime_complete"].apply(lambda x: True if x else False)
        run_time = df["run_time"][completed_runs].apply(lambda x: x.total_seconds() / 3600.0)
        logger.info(f"\tTotal study simulation run time: {run_time.sum():.4f} hrs")
        logger.info(f"\tAverage trial simulation run time: {run_time.mean():.4f} hrs")
    
    if (total_completed_trials < n_trials) and (total_completed_trials > 1):
    
        trails_remaining = n_trials - total_completed_trials
        time_needed = trails_remaining * run_time.mean()
        logger.info(
            f"\tEstimated remaining simulation time needed: {time_needed:.4f} hrs")

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
                    f"\tWith a given wall-time of {time_str}, submit {nodes} PBS workers to complete the study")
           
        if "slurm" in hyper_config:
            time_str =  hyper_config["slurm"]["batch"]["t"]
            if ":" in time_str:
                walltime = get_sec(time_str)
            else:
                walltime = False
            if walltime:
                nodes = int(np.ceil(3600 * time_needed / walltime))
                logger.info(
                    f"\tWith a given wall-time of {time_str}, submit {nodes} SLURM workers to complete the study")   

    return total_completed_trials
