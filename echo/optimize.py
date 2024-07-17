from echo.src.config import (
    config_check,
    recursive_config_reader,
    configure_storage,
    configure_sampler,
    configure_pruner,
)
from echo.src.reporting import study_report
from argparse import ArgumentParser
import subprocess
import logging
import optuna
import shutil
import yaml
import sys
import os
import warnings

from typing import List

warnings.filterwarnings("ignore")


def args():
    parser = ArgumentParser(
        description="ECHO: A distributed multi-gpu hyperparameter optimization package build with Optuna"
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
        "--study_name",
        dest="study_name",
        type=str,
        default=False,
        help="The name of the study",
    )
    parser.add_argument(
        "--storage_type",
        dest="storage_type",
        type=str,
        default=False,
        help="The storage type (sqlite or maria)",
    )
    parser.add_argument(
        "--storage",
        dest="storage",
        type=str,
        default=False,
        help="The storage name of the database to use",
    )
    parser.add_argument(
        "--delete_study",
        dest="delete_study",
        type=bool,
        default=False,
        help="Delete the study from the storage db",
    )
    parser.add_argument(
        "-o",
        "--objective",
        dest="objective",
        type=str,
        default=False,
        help="Path to the supplied objective class",
    )
    parser.add_argument(
        "-d",
        "--direction",
        dest="direction",
        type=str,
        default=False,
        help="Direction of the metric. Choose from maximize or minimize",
    )
    parser.add_argument(
        "-m",
        "--metric",
        dest="metric",
        type=str,
        default=False,
        help="The validation metric",
    )
    parser.add_argument(
        "-t",
        "--trials",
        dest="n_trials",
        type=str,
        default=False,
        help="The number of trials in the study",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        dest="gpu",
        type=str,
        default=False,
        help="Use the gpu or not (bool)",
    )
    parser.add_argument(
        "-s",
        "--save_path",
        dest="save_path",
        type=str,
        default=False,
        help="Path to the save directory",
    )
    parser.add_argument(
        "-c",
        "--create_study",
        dest="create_study",
        type=str,
        default=False,
        help="Create a study but do not submit any workers",
    )

    return vars(parser.parse_args())


def fix_broken_study(
    _study: optuna.study.Study,
    name: str,
    storage: str,
    direction: str,
    sampler: optuna.samplers.BaseSampler,
    pruner: optuna.pruners.NopPruner,
) -> (optuna.study.Study, List[optuna.trial.Trial]):

    """
    This method removes broken trials, which are those
    that failed to complete 1 epoch before slurm (or something else) killed the job
    and returned NAN or NONE.

    Failure to remove these trails leads to a error when optuna tries to update the
    parameters. This is because these trails only have "NoneType" data associated
    with them, but we need numerical data (e.g. the loss value) to update parameters.
    """

    if len(_study.trials) == 0:
        return _study, []

    trials = []
    removed = []
    for trial in _study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trials.append(trial)
            continue
        if len(trial.intermediate_values) == 0:
            trials.append(trial)
            continue
        step, intermediate_value = max(trial.intermediate_values.items())
        # and (np.isfinite(intermediate_value)):
        if intermediate_value is not None:
            trials.append(trial)
        else:
            removed.append(trial.number + 1)

    if len(removed) == 0:
        return _study, []

    """ Delete the current study """
    optuna.delete_study(study_name=name, storage=storage)

    """ Create a new one in its place """
    if isinstance(direction, str):
        study_fixed = optuna.create_study(
            study_name=name,
            storage=storage,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=False,
        )
    else:
        study_fixed = optuna.create_study(
            study_name=name,
            storage=storage,
            sampler=sampler,
            directions=direction,
            load_if_exists=True,
        )

    """ Add the working trials to the new study """
    for trial in trials:
        study_fixed.add_trial(trial)

    return study_fixed, removed



def generate_batch_commands(
    hyper_config, batch_type, aiml_path, jobid, batch_commands = []
) -> List[str]:

    # Check if "gpus_per_node" is specified in hyper_config[batch_type]
    if "gpus_per_node" in hyper_config[batch_type]:
        # Get the list of GPU devices, or convert a single integer to a list
        gpus_per_node = list(range(hyper_config[batch_type]["gpus_per_node"]))
        
        # Check if "tasks_per_worker" is specified in hyper_config[batch_type]
        if (
            "tasks_per_worker" in hyper_config[batch_type]
            and hyper_config[batch_type]["tasks_per_worker"] > 1
        ):
            # Warn about the experimental nature of tasks_per_worker
            logging.warning(
                "The tasks_per_worker is experimental; be advised that some runs may fail."
            )
            logging.warning(
                "Check the log and stdout/err files if simulations are dying to see the errors."
            )

            # Loop over the specified number of trials
            for copy in range(hyper_config[batch_type]["tasks_per_worker"]):
                # Loop over each GPU device
                for device in gpus_per_node:
                    # Append the command with CUDA_VISIBLE_DEVICES={device} to batch_commands
                    batch_commands.append(
                        f"CUDA_VISIBLE_DEVICES={device}, {aiml_path} -n {jobid} &"
                    )
                # Allow some time between calling instances of run
                batch_commands.append("sleep 0.5")
            # Wait for all background jobs to finish
            batch_commands.append("wait")
        else:
            # Loop over each GPU device without multiple trials
            for device in gpus_per_node:
                # Append the command with CUDA_VISIBLE_DEVICES={device} to batch_commands
                batch_commands.append(
                    f"CUDA_VISIBLE_DEVICES={device}, {aiml_path} -n {jobid} &"
                )
            batch_commands.append("wait")
    elif (
        "tasks_per_worker" in hyper_config[batch_type]
        and hyper_config[batch_type]["tasks_per_worker"] > 1
    ):
        # Warn about the experimental nature of tasks_per_worker
        logging.warning(
            "The trails_per_job is experimental, be advised that some runs may fail."
        )
        logging.warning(
            "Check the log and stdout/err files if simulations are dying to see the errors."
        )
        # Loop over the specified number of trials
        for copy in range(hyper_config[batch_type]["tasks_per_worker"]):
            # Append the command to batch_commands
            batch_commands.append(
                f"{aiml_path} -n {jobid} &"
            )
            # Allow some time between calling instances of run
            batch_commands.append("sleep 0.5")
        # Wait for all background jobs to finish
        batch_commands.append("wait")
    else:
        # Append the default command to batch_commands
        batch_commands.append(f"{aiml_path} -n {jobid}")

    return batch_commands


def prepare_slurm_launch_script(hyper_config: str, model_config: str) -> List[str]:

    slurm_options = ["#!/bin/bash -l"]
    slurm_options += [
        f"#SBATCH -{arg} {val}" if len(arg) == 1 else f"#SBATCH --{arg}={val}"
        for arg, val in hyper_config["slurm"]["batch"].items()
    ]  # This needs updated to redirect the slurm output to the save_path
    if "bash" in hyper_config["slurm"]:
        if len(hyper_config["slurm"]["bash"]) > 0:
            for line in hyper_config["slurm"]["bash"]:
                slurm_options.append(line)
    if "kernel" in hyper_config["slurm"]:
        if hyper_config["slurm"]["kernel"] is not None:
            slurm_options.append(f'{hyper_config["slurm"]["kernel"]}')
    hyper_base = os.path.basename(sys.argv[1])
    model_base = os.path.basename(sys.argv[2])
    aiml_path = f"echo-run {hyper_base} {model_base}"
    slurm_id = "$SLURM_JOB_ID"
    # hyper_config, batch_type, aiml_path, jobid, batch_commands = []
    return generate_batch_commands(hyper_config, "slurm", aiml_path, slurm_id, batch_commands=slurm_options)


def prepare_pbs_launch_script(hyper_config: str, model_config: str) -> List[str]:
    pbs_options = ["#!/bin/bash -l"]
    for arg, val in hyper_config["pbs"]["batch"].items():
        if arg == "l" and type(val) == list:
            for opt in val:
                pbs_options.append(f"#PBS -{arg} {opt}")
        elif len(arg) == 1:
            if arg in ["o", "e"]:
                if hyper_config["save_path"] in val:
                    pbs_options.append(f"#PBS -{arg} {val}")
                elif hyper_config["save_path"] not in val:
                    pbs_options.append(f"#PBS -{arg} {hyper_config['save_path']+val}")
            else:
                pbs_options.append(f"#PBS -{arg} {val}")
        elif arg in ["o", "e"]:
            if val != "/dev/null":
                _val = os.path.append(hyper_config["save_path"], val)
                # info?
                pbs_options.append(f"#PBS -{arg} {_val}")
            else:
                pbs_options.append(f"#PBS -{arg} {val}")
        else:
            pbs_options.append(f"#PBS --{arg}={val}")
    if "bash" in hyper_config["pbs"]:
        if len(hyper_config["pbs"]["bash"]) > 0:
            for line in hyper_config["pbs"]["bash"]:
                pbs_options.append(line)
    if "kernel" in hyper_config["pbs"]:
        if hyper_config["pbs"]["kernel"] is not None:
            pbs_options.append(f'{hyper_config["pbs"]["kernel"]}')
    hyper_base = os.path.basename(sys.argv[1])
    model_base = os.path.basename(sys.argv[2])
    aiml_path = f"echo-run {hyper_base} {model_base}"
    pbs_jobid = "$PBS_JOBID"
    return generate_batch_commands(hyper_config, "pbs", aiml_path, pbs_jobid, batch_commands=pbs_options)


def main():

    args_dict = args()

    _hyper_config = args_dict.pop("hyperparameter")
    _model_config = args_dict.pop("model")

    assert (
        _hyper_config and _model_config
    ), "Usage: python main.py hyperparameter.yml model.yml [optional parser options]"

    assert os.path.isfile(
        _hyper_config
    ), f"Hyperparameter optimization config file {_hyper_config} does not exist"
    with open(_hyper_config) as f:
        hyper_config = yaml.load(f, Loader=yaml.FullLoader)

    assert os.path.isfile(
        _model_config
    ), f"Model config file {_model_config} does not exist"
    with open(_model_config) as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)

    """ Set up a logger """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    """ Stream output to stdout"""
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    """ Override other options in hyperparameter config file, if supplied """
    for name, val in args_dict.items():
        if val and (name in hyper_config):
            if name == "save_path":
                current_value = hyper_config[name]
                logging.info(
                    f"Over-riding {name} in the hyperparameter configuration: {current_value} -> {val}"
                )
                hyper_config["save_path"] = val
            else:
                current_value = hyper_config["optuna"][name]
                logging.info(
                    f"Over-riding {name} in the hyperparameter configuration: {current_value} -> {val}"
                )
                hyper_config["optuna"][name] = val

    """ Run some tests on the configurations """
    config_check(hyper_config, model_config)

    """ Global save path """
    save_path = hyper_config["save_path"]

    """ Create the save directory if it does not already exist """
    if os.path.isdir(save_path):
        logging.info(f"The parent save_path already exists at {save_path}")
    else:
        logging.info(f"Creating parent save_path at {save_path}")
        os.makedirs(save_path, exist_ok=True)

    """ Save the config files to the save_path """
    for fn in [_hyper_config, _model_config]:
        fn_base = os.path.basename(fn)
        if not os.path.isfile(os.path.join(save_path, fn_base)):
            shutil.copyfile(fn, os.path.join(save_path, fn_base))

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

    """ Print the configurations to the logger """
    logging.info("Current hyperparameter configuration settings:")
    for p, v in recursive_config_reader(hyper_config):
        full_path = ".".join([str(_p) for _p in p])
        logging.info(f"\t{full_path}: {v}")
    logging.info("Current model configuration settings:")
    for p, v in recursive_config_reader(model_config):
        full_path = ".".join([str(_p) for _p in p])
        logging.info(f"\t{full_path}: {v}")

    """ Print the configurations to the logger """
    study_name = hyper_config["optuna"]["study_name"]

    """ Set up storage db """
    storage = configure_storage(hyper_config)

    """ Initialize the sampler """
    sampler = configure_sampler(hyper_config)

    """ Initialize the pruner """
    pruner = configure_pruner(hyper_config)

    """ Initialize study direction(s) """
    direction = hyper_config["optuna"]["direction"]
    single_objective = isinstance(direction, str)

    """
        Check if the db entry exists already
    """
    try:
        """Check if the study record already exists."""
        optuna.load_study(study_name=study_name, storage=storage)

        if args_dict["delete_study"]:
            logging.info(
                f"Removing the study_name '{study_name}' that exists in storage {storage}."
            )
            optuna.delete_study(study_name=study_name, storage=storage)

            reload_study = False
        else:
            logging.warning(f"The study '{study_name}' exists in {storage}.")
            logging.info(
                f"\tIf you want to delete the study '{study_name}' first, pass '--delete_study 1'."
            )
            reload_study = True

    except KeyError:
        """The study name was not in storage, can proceed"""
        reload_study = False

    """
        Initiate a study for the first time
    """
    if not reload_study:
        """Check the direction"""
        if isinstance(direction, list):
            for direc in direction:
                if direc not in ["maximize", "minimize"]:
                    raise OSError(
                        f"Optimizer direction {direc} not recognized. Choose from maximize or minimize"
                    )

        else:
            if direction not in ["maximize", "minimize"]:
                raise OSError(
                    f"Optimizer direction {direction} not recognized. Choose from maximize or minimize"
                )

        """ Create a new study in the storage object """
        if single_objective:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction=direction,
                sampler=sampler,
                pruner=pruner,
            )
        else:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                sampler=sampler,
                directions=direction,
            )

    else:
        """Check to see if there are any broken trials"""

        logging.info(
            "Checking the study for broken trials (those that did not complete 1 epoch before dying)"
        )
        if single_objective:
            study = optuna.load_study(
                study_name=study_name,
                storage=storage,
                sampler=sampler,
                pruner=pruner,
            )
        else:
            study = optuna.load_study(
                study_name=study_name,
                storage=storage,
                sampler=sampler,
            )
        study, removed = fix_broken_study(
            study, study_name, storage, direction, sampler, pruner
        )

        if len(removed):
            logging.info(f"\tRemoving problematic trials {removed}.")
        else:
            logging.info("\tAll trials check out!")

        """ Report on the current study to the logger """
        n_trials = hyper_config["optuna"]["n_trials"]
        total_comp = study_report(study, hyper_config)
        if total_comp >= n_trials:
            logging.warning(
                "The number of trials in the study equals or exceeds that requested. Exiting without error."
            )
            sys.exit()

    # Check if the study is empty and the 'enqueue' option is provided
    if len(hyper_config['optuna'].get('enqueue', [])) and len(study.trials) == 0:
        for params in hyper_config['optuna']['enqueue']:
            logging.info(f"Adding trial parameters {params} to the study")
            study.enqueue_trial(params, skip_if_exists=True)

    """
        Override to create the database but skip submitting jobs.
    """
    create_db_only = True if args_dict["create_study"] else False

    # Stop here if arg is defined -- intention is that you manually run echo-run (run.py) for debugging purposes
    if create_db_only:
        logging.info(f"Created study {study_name} located at {storage}. Exiting.")
        sys.exit()

    current_directory = sys.path[0]
    working_directory = save_path
    os.chdir(working_directory)

    """ SLURM SUPPORT """

    """ Prepare launch script """
    if "slurm" in hyper_config:
        launch_script = prepare_slurm_launch_script(hyper_config, model_config)

        """ Save the configured script """
        script_location = os.path.join(save_path, "launch_slurm.sh")
        with open(script_location, "w") as fid:
            for line in launch_script:
                fid.write(f"{line}\n")

        """ Launch the slurm jobs """
        job_ids = []
        name_condition = "J" in hyper_config["slurm"]["batch"]
        slurm_job_name = (
            hyper_config["slurm"]["batch"]["J"] if name_condition else "echo_trial"
        )
        n_workers = hyper_config["slurm"]["jobs"]
        for worker in range(n_workers):
            w = subprocess.Popen(
                f"sbatch -J {slurm_job_name}_{worker} {script_location}",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).communicate()
            job_ids.append(w[0].decode("utf-8").strip("\n").split(" ")[-1])
            logging.info(
                f"Submitted slurm batch job {worker + 1}/{n_workers} with id {job_ids[-1]}"
            )

        """ Write the job ids to file for reference """
        with open(os.path.join(save_path, "slurm_job_ids.txt"), "w") as fid:
            for line in job_ids:
                fid.write(f"{line}\n")

    """ PBS SUPPORT """

    if "pbs" in hyper_config:
        launch_script = prepare_pbs_launch_script(hyper_config, model_config)

        """ Save the configured script """
        script_location = os.path.join(save_path, "launch_pbs.sh")
        with open(script_location, "w") as fid:
            for line in launch_script:
                fid.write(f"{line}\n")
                
        """ Launch the slurm jobs """
        job_ids = []
        name_condition = "N" in hyper_config["pbs"]["batch"]
        slurm_job_name = (
            hyper_config["pbs"]["batch"]["N"] if name_condition else "echo_trial"
        )
        n_workers = hyper_config["pbs"]["jobs"]
        for worker in range(n_workers):
            w = subprocess.Popen(
                f"qsub -N {slurm_job_name}_{worker} {script_location}",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).communicate()
            job_ids.append(w[0].decode("utf-8").strip("\n"))
            logging.info(
                f"Submitted pbs batch job {worker + 1}/{n_workers} with id {job_ids[-1]}"
            )

        """ Write the job ids to file for reference """
        with open(os.path.join(save_path, "pbs_job_ids.txt"), "w") as fid:
            for line in job_ids:
                fid.write(f"{line}\n")

    os.chdir(current_directory)


if __name__ == "__main__":
    main()
