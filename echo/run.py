import os
import yaml
import time
import optuna
import logging
import importlib.machinery
from argparse import ArgumentParser

from echo.src.config import (
    config_check,
    configure_storage,
    configure_sampler,
    configure_pruner,
)
from echo.src.reporting import successful_trials, get_sec, devices
import warnings

warnings.filterwarnings("ignore")


# References
# https://github.com/optuna/optuna/issues/1365
# https://docs.dask.org/en/latest/setup/hpc.html
# https://dask-cuda.readthedocs.io/en/latest/worker.html
# https://optuna.readthedocs.io/en/stable/tutorial/004_distributed.html#distributed


start_the_clock = time.time()


def args():
    parser = ArgumentParser(
        description="echo-run: A distributed multi-gpu hyperparameter optimization package build with Optuna"
    )

    parser.add_argument(
        "hyperparameter",
        type=str,
        help="Path to the hyperparameter configuration containing your inputs.",
    )

    parser.add_argument(
        "model",
        type=str,
        help="Path to the model configuration containing your inputs.",
    )

    parser.add_argument(
        "-n",
        dest="node_id",
        type=str,
        help="PBS/SLURM job name/identification (default = None)",
        default=None,
    )

    parser.add_argument(
        "-w",
        dest="wall_time",
        type=str,
        default="12:00:00",
        help="Set the maximum running time in HH:MM:SS. (default = 12:00:00)",
    )

    return vars(parser.parse_args())


def main():

    args_dict = args()

    hyper_fn = args_dict.pop("hyperparameter")
    model_fn = args_dict.pop("model")
    node_id = args_dict.pop("node_id")

    assert hyper_fn and model_fn, "Usage: python run.py hyperparameter.yml model.yml"

    """ Set up a logger """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    """ Stream output to stdout """
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    """ Run some tests on the configurations """
    config_check(hyper_fn, model_fn, file_check=True)

    """ Load config files """
    with open(hyper_fn) as f:
        hyper_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(model_fn) as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)

    """ Get the path to save all the data """
    save_path = hyper_config["save_path"]
    logging.info(f"Saving trial details to {save_path}")

    """ Create the save directory if it does not already exist """
    if not os.path.isdir(save_path):
        logging.info(f"Creating parent save_path at {save_path}")
        os.makedirs(save_path, exist_ok=True)

    """ Stream output to file """
    _log = False if "log" not in hyper_config else hyper_config["log"]
    if _log:
        fh = logging.FileHandler(
            os.path.join(save_path, "log.txt"),
            mode="a+",  # always initiate / append
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    """ Print job id to the logger """
    if node_id is not None:
        logging.info(f"Running on PBS/SLURM batch id: {node_id}")
    else:
        logging.info("Running as __main__")

    """ Copy the optuna details to the model config """
    model_config["optuna"] = hyper_config["optuna"]
    model_config["optuna"]["save_path"] = hyper_config["save_path"]

    """ Import user-supplied Objective class """
    logging.info(
        f"Importing custom objective from {model_config['optuna']['objective']}"
    )
    loader = importlib.machinery.SourceFileLoader(
        "custom_objective", model_config["optuna"]["objective"]
    )
    mod = loader.load_module()
    from custom_objective import Objective

    """ Obtain GPU/CPU ids """
    device = devices(model_config["optuna"]["gpu"])

    """ Initialize the study object """
    study_name = model_config["optuna"]["study_name"]

    """ Set up storage db """
    storage = configure_storage(hyper_config)

    """ Initialize the sampler """
    sampler = configure_sampler(hyper_config)

    """  Initialize the pruner """
    pruner = configure_pruner(hyper_config)

    """ Initialize study direction(s) """
    direction = model_config["optuna"]["direction"]
    single_objective = isinstance(direction, str)
    logging.info(f"Direction of optimization: {direction}")

    """ Initialize the optimization metric(s) """
    if isinstance(model_config["optuna"]["metric"], list):
        metric = [str(m) for m in model_config["optuna"]["metric"]]
    else:
        metric = str(model_config["optuna"]["metric"])
    logging.info(f"Using metric {metric}")

    """ Load or initiate study """
    if single_objective:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            direction=direction,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            directions=direction,
            load_if_exists=True,
        )
    logging.info(f"Loaded study {study_name} located at {storage}")

    # Check if the study is empty and the 'enqueue' option is provided
    if len(hyper_config['optuna'].get('enqueue', [])) and len(study.trials) == 0:
        for params in hyper_config['optuna']['enqueue']:
            logging.info(f"Adding trial parameters to the study {params}")
            study.enqueue_trial(params, skip_if_exists=True)

    # Check here if 'trial_results' directory exists, which should be created by optimize.py
    # The reason for it being here is if you are running in debug mode 
    trial_results_path = os.path.join(hyper_config['save_path'], "trial_results")
    if not os.path.isdir(trial_results_path):
        os.makedirs(trial_results_path, exist_ok=True)

    """ Initialize objective function """
    objective = Objective(model_config, metric)
    objective.set_properties(node_id=node_id, device=device)

    """ Optimize it """
    logging.info(
        f'Running optimization for {model_config["optuna"]["n_trials"]} trials'
    )

    """ Get the cluster job wall-time """
    if "slurm" in hyper_config:
        wall_time = hyper_config["slurm"]["batch"]["t"]
        logging.info(f"Running trials for a maximum slurm wall-time {wall_time}")
    elif "pbs" in hyper_config:
        wall_time = False
        for option in hyper_config["pbs"]["batch"]["l"]:
            if "walltime" in option:
                wall_time = option.split("walltime=")[-1]
                break
        if wall_time is False:
            logging.warning(
                "Could not process the walltime for run.py. Assuming 12 hours."
            )
            wall_time = args_dict.pop("wall_time")
        logging.info(f"Running trials for a maximum PBS wall-time {wall_time}")
    else:
        wall_time = args_dict.pop("wall_time")
        logging.info(f"Running trials as main for default wall-time of {wall_time}")
        logging.info("The wall-time is controlled by the -w option. See --help.")
    wall_time_secs = get_sec(wall_time)

    logging.warning("Attempting to run trials and stop before hitting the wall-time")
    logging.warning(
        "Some trials may not complete if the wall-time is reached. Optuna will start over."
    )

    estimated_run_time = wall_time_secs - (time.time() - start_the_clock)
    while successful_trials(study) < model_config["optuna"]["n_trials"]:
        try:
            study.optimize(
                objective,
                n_trials=1,
                timeout=estimated_run_time,
                # catch = (ValueError,) # Later to be added as a config option
            )
        except KeyboardInterrupt:
            logging.warning("Recieved signal to die from keyboard. Exiting.")
            break
        except Exception as E:
            logging.warning(f"Died due to due to error {E}")
            break

        """ Early stopping if too close to the wall time """
        df = study.trials_dataframe()
        if df.shape[0] > 1:
            df["run_time"] = df["datetime_complete"] - df["datetime_start"]
            completed_runs = df["datetime_complete"].apply(
                lambda x: True if x else False
            )
            run_times = df["run_time"][completed_runs].apply(
                lambda x: x.total_seconds()
            )
            max_run_time = run_times.max()
            time_left = wall_time_secs - (time.time() - start_the_clock)

            if max_run_time >= time_left:
                logging.warning(
                    "Stopping early since the longest observed run-time in the study exceeds the time remaining on this node."
                )
                break

        else:  # no trials in the database yet
            time_left = wall_time_secs - (time.time() - start_the_clock)

            if time_left < (
                wall_time_secs / 2
            ):  # if more than half the time remaining, launch another trial
                logging.warning(
                    "Stopping early since the longest observed run-time in the study exceeds the time remaining on this node."
                )
                break

        """ Update the study optimizer timeout"""
        time_left = wall_time_secs - (time.time() - start_the_clock)
        estimated_run_time = 0.95 * time_left


if __name__ == "__main__":
    main()
