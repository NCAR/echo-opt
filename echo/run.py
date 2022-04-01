import os
import sys
import yaml
import glob
import time
import optuna
import logging
import subprocess
import numpy as np
import pandas as pd
import importlib.machinery
from echo.src.pruners import pruners
from echo.src.samplers import samplers
import warnings
warnings.filterwarnings("ignore")


# References
# https://github.com/optuna/optuna/issues/1365
# https://docs.dask.org/en/latest/setup/hpc.html
# https://dask-cuda.readthedocs.io/en/latest/worker.html
# https://optuna.readthedocs.io/en/stable/tutorial/004_distributed.html#distributed


start_the_clock = time.time()


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


def get_sec(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


def main():

    if len(sys.argv) != 3:
        print(
            "Usage: python run.py hyperparameter.yml model.yml"
        )
        sys.exit()

    hyper_fn = str(sys.argv[1])
    model_fn = str(sys.argv[2])

    # Set up a logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    ################################################################

    # Check if hyperparameter config file exists
    assert os.path.isfile(
        hyper_fn),  f"Hyperparameter optimization config file {hyper_fn} does not exist"
    with open(hyper_fn) as f:
        hyper_config = yaml.load(f, Loader=yaml.FullLoader)

    # Check if the wall-time exists
    if "slurm" in hyper_config:
        assert "t" in hyper_config["slurm"]["batch"],  "You must supply a wall time in the hyperparameter config at slurm:batch:t"

    if "pbs" in hyper_config:
        assert any([("walltime" in x) for x in hyper_config["pbs"]["batch"]["l"]]
                   ), "You must supply a wall time in the hyperparameter config at pbs:bash:l"

    # Check if model config file exists
    assert os.path.isfile(
        model_fn), f"Model config file {model_fn} does not exist"
    with open(model_fn) as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)

    # Copy the optuna details to the model config
    model_config["optuna"] = hyper_config["optuna"]

    # Check if path to objective method exists
    assert os.path.isfile(model_config["optuna"]["objective"]
                          ), f'The objective file {model_config["optuna"]["objective"]} does not exist'

    loader = importlib.machinery.SourceFileLoader(
        "custom_objective",
        model_config["optuna"]["objective"]
    )
    mod = loader.load_module()
    from custom_objective import Objective

    # Check if the optimization metric direction is supported
    direction = model_config["optuna"]["direction"]
    single_objective = isinstance(direction, str)

    if single_objective:
        assert direction in [
            "maximize", "minimize"], f"Optimizer direction {direction} not recognized. Choose from maximize or minimize"
    else:
        for direc in direction:
            assert direc in [
                "maximize", "minimize"], f"Optimizer direction {direc} not recognized. Choose from maximize or minimize"

    logging.info(f"Direction of optimization: {direction}")

    # Add other config checks

    ################################################################

    # Stream output to log file
    if "log" in hyper_config:
        savepath = hyper_config["log"]["save_path"] if "save_path" in hyper_config["log"] else "log.txt"
        mode = "a+" if bool(hyper_config["optuna"]["reload"]) else "w"
        fh = logging.FileHandler(savepath,
                                 mode=mode,
                                 encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # Get the path to save all the data
    save_path = model_config["optuna"]["save_path"]
    logging.info(f"Saving optimization details to {save_path}")

    # Grab the metric
    if isinstance(model_config["optuna"]["metric"], list):
        metric = [str(m) for m in model_config["optuna"]["metric"]]
    else:
        metric = str(model_config["optuna"]["metric"])
    logging.info(f"Using metric {metric}")

    # Get list of devices and initialize the Objective class
    if bool(model_config["optuna"]["gpu"]):
        try:
            gpu_report = sorted(
                gpu_report().items(),
                key=lambda x: x[1],
                reverse=True
            )
            device = gpu_report[0][1]
        except:
            logging.warning(
                "The gpu is not responding to a call from nvidia-smi.\
                Setting gpu device = 0, but this may fail."
            )
            device = 0
    else:
        device = 'cpu'
    logging.info(f"Using device {device}")

    ################################################################

    # Initialize the study object
    study_name = model_config["optuna"]["study_name"]
    reload_study = bool(model_config["optuna"]["reload"])

    # Identify the storage location
    storage = model_config["optuna"]["storage"]

    # Initialize the sampler
    if "sampler" not in hyper_config["optuna"]:
        if single_objective:  # single-objective
            sampler = optuna.samplers.TPESampler()
        else:  # multi-objective equivalent of TPESampler
            sampler = optuna.multi_objective.samplers.MOTPEMultiObjectiveSampler()
    else:
        sampler = samplers(hyper_config["optuna"]["sampler"])

    if "pruner" not in hyper_config["optuna"]:
        pruner = optuna.pruners.NopPruner()
    else:
        pruner = pruners(hyper_config["optuna"]["pruner"])

    # Load or initiate study
    if single_objective:
        study = optuna.create_study(study_name=study_name,
                                    storage=storage,
                                    sampler=sampler,
                                    pruner=pruner,
                                    direction=direction,
                                    load_if_exists=True)
    else:
        study = optuna.multi_objective.study.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            directions=direction,
            load_if_exists=True
        )
    logging.info(f"Loaded study {study_name} located at {storage}")

    # Initialize objective function
    objective = Objective(model_config, metric, device)

    # Optimize it
    logging.info(
        f'Running optimization for {model_config["optuna"]["n_trials"]} trials'
    )

    # Get the cluster job wall-time
    if "slurm" in hyper_config:
        wall_time = hyper_config["slurm"]["batch"]["t"]
    elif "pbs" in hyper_config:
        wall_time = False
        for option in hyper_config["pbs"]["batch"]["l"]:
            if "walltime" in option:
                wall_time = option.split("walltime=")[-1]
                break
        if wall_time is False:
            logging.warning(
                "Could not process the walltime for run.py. Assuming 12 hours.")
            wall_time = "12:00:00"
    wall_time_secs = get_sec(wall_time)

    logging.info(
        f"This script will run for a fraction of the wall-time of {wall_time} and try to die without error"
    )

    run_times = []
    estimated_run_time = wall_time_secs
    for iteration in range(int(model_config["optuna"]["n_trials"])):
        try:
            start_time = time.time()
            study.optimize(
                objective,
                n_trials=1,
                timeout=estimated_run_time,
                #catch = (ValueError,)
            )
            end_time = time.time()
            run_times.append(end_time - start_time)
        except KeyboardInterrupt:
            logging.warning(
                f"Recieved signal to die from keyboard. Exiting."
            )
            break
        except Exception as E:
            logging.warning(
                f"Dying early due to error {E}"
            )
            break

        # Testing out way to stop running trials if too close to wall-time.
        # Update to computing the mean of the run times of all completed trials in the database.
        if len(run_times) > 1:
            average_run_time = np.mean(run_times)
            sigma_run_time = np.std(run_times) if len(run_times) > 2 else 0.0
            estimated_run_time = average_run_time + 2 * sigma_run_time
            time_left = wall_time_secs - (time.time() - start_the_clock)
            if time_left < estimated_run_time:
                logging.warning(
                    f"Dying early as estimated run-time exceeds the time remaining on this node."
                )
                break


if __name__ == "__main__":
    main()
