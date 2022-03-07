import warnings
warnings.filterwarnings("ignore")

import os
import sys
import yaml
import optuna
import logging
import subprocess
import numpy as np
from argparse import ArgumentParser
from echo.src.samplers import samplers
from echo.src.pruners import pruners
from typing import Dict


def args():
    parser = ArgumentParser(description=
        "ECHO: A distributed multi-gpu hyperparameter optimization package build with Optuna"
    )

    parser.add_argument("hyperparameter", type=str, help=           
            "Path to the hyperparameter configuration containing your inputs."
    )
    
    parser.add_argument("model", type=str, help=
            "Path to the model configuration containing your inputs."
    )
    parser.add_argument(
        "-n",
        "--study_name", 
        dest="study_name", 
        type=str,
        default=False, 
        help="The name of the study"
    )
    parser.add_argument(
        "--override", 
        dest="override", 
        type=bool,
        default=False,
        help="Force remove the study name from the storage"
    )
    parser.add_argument(
        "-r", 
        "--reload", 
        dest="reload", 
        type=str,
        default=False, 
        help="Set = 0 to initiate a new study, = 1 to continue a study"
    )
    parser.add_argument(
        "-o", 
        "--objective", 
        dest="objective", 
        type=str,
        default=False, 
        help="Path to the supplied objective class"
    )
    parser.add_argument(
        "-d", 
        "--direction", 
        dest="direction", 
        type=str,
        default=False, 
        help="Direction of the metric. Choose from maximize or minimize"
    )
    parser.add_argument(
        "-m", 
        "--metric", 
        dest="metric", 
        type=str,
        default=False, 
        help="The validation metric"
    )
    parser.add_argument(
        "-t", 
        "--trials", 
        dest="n_trials", 
        type=str,
        default=False, 
        help="The number of trials in the study"
    )
    parser.add_argument(
        "-g", 
        "--gpu", 
        dest="gpu", 
        type=str,
        default=False, 
        help="Use the gpu or not (bool)"
    )
    parser.add_argument(
        "-s", 
        "--save_path", 
        dest="save_path", 
        type=str,
        default=False, 
        help="Path to the save directory"
    )   
    parser.add_argument(
        "-c", 
        "--create_study", 
        dest="create_study", 
        type=str,
        default=False, 
        help="Create a study but do not submit any workers"
    )   
    return vars(parser.parse_args())


def fix_broken_study(_study: optuna.study.Study, 
                     name: str, 
                     storage: str, 
                     direction: str, 
                     sampler: optuna.samplers.BaseSampler, 
                     pruner: optuna.pruners.NopPruner):
    
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
        if (intermediate_value is not None): # and (np.isfinite(intermediate_value)):
            trials.append(trial)
        else:
            removed.append(trial.number+1)
            
    if len(removed) == 0:
        return _study, []
    
    # Delete the current study
    optuna.delete_study(study_name=name, storage=storage)
    
    # Create a new one in its place
    if isinstance(direction, str):
        study_fixed = optuna.create_study(study_name=name, 
                                      storage=storage, 
                                      direction=direction,
                                      sampler=sampler,
                                          pruner=pruner,
                                      load_if_exists=False)
    else:
        study_fixed = optuna.multi_objective.create_study(
            study_name=name,
            storage=storage,
            directions=direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=False
        )
    
    # Add the working trials to the new study
    for trial in trials:
        study_fixed.add_trial(trial)
        
    return study_fixed, removed


def prepare_slurm_launch_script(hyper_config: str, 
                                model_config: str):
    
    slurm_options = ["#!/bin/bash -l"]
    slurm_options += [
        f"#SBATCH -{arg} {val}" if len(arg) == 1 else f"#SBATCH --{arg}={val}" 
        for arg, val in hyper_config["slurm"]["batch"].items()
    ]
    if "bash" in hyper_config["slurm"]:
        if len(hyper_config["slurm"]["bash"]) > 0:
            for line in hyper_config["slurm"]["bash"]:
                slurm_options.append(line)
    if "kernel" in hyper_config["slurm"]:
        if hyper_config["slurm"]["kernel"] is not None:
            slurm_options.append(f'{hyper_config["slurm"]["kernel"]}')
    import aimlutils.echo as opt
    aiml_path = os.path.join(
        os.path.abspath(opt.__file__).strip("__init__.py"), 
        "run.py"
    )
    if "trials_per_job" in hyper_config["slurm"]:
        for copy in range(hyper_config["slurm"]["trials_per_job"]):
            slurm_options.append(f"python {aiml_path} {sys.argv[1]} {sys.argv[2]} &")
            slurm_options.append(f"sleep 30") # allow some time between calling instances of run
        slurm_options.append(f"wait")
    else:
        slurm_options.append(f"python {aiml_path} {sys.argv[1]} {sys.argv[2]}")
    return slurm_options


def prepare_pbs_launch_script(hyper_config: str,
                              model_config: str):
    
    pbs_options = ["#!/bin/bash -l"]
    for arg, val in hyper_config["pbs"]["batch"].items():
        if arg == "l" and type(val) == list:
            for opt in val:
                pbs_options.append(f"#PBS -{arg} {opt}")
        elif len(arg) == 1:
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
    import aimlutils.echo as opt
    aiml_path = os.path.join(
        os.path.abspath(opt.__file__).strip("__init__.py"), 
        "run.py"
    )
    if "trials_per_job" in hyper_config["pbs"]:
        for copy in range(hyper_config["pbs"]["trials_per_job"]):
            pbs_options.append(f"python {aiml_path} {sys.argv[1]} {sys.argv[2]} &")
            pbs_options.append(f"sleep 30") # allow some time between calling instances of run
        pbs_options.append(f"wait")
    else:
        pbs_options.append(f"python {aiml_path} {sys.argv[1]} {sys.argv[2]}")
    return pbs_options


def recursive_config_reader(_dict: Dict[str, str], 
                         path: bool = None):
    
    if path is None:
        path = []
    for k,v in _dict.items():
        newpath = path + [k]
        if isinstance(v, dict):
            for u in recursive_config_reader(v, newpath):
                yield u
        else:
            yield newpath, v


def main():
    
    args_dict = args()

    hyper_config = args_dict.pop("hyperparameter")
    model_config = args_dict.pop("model")

    assert (hyper_config and model_config), "Usage: python main.py hyperparameter.yml model.yml [optional parser options]"

    assert os.path.isfile(hyper_config), f"Hyperparameter optimization config file {hyper_config} does not exist"
    with open(hyper_config) as f:
        hyper_config = yaml.load(f, Loader=yaml.FullLoader)

    assert os.path.isfile(model_config), f"Model config file {model_config} does not exist"
    with open(model_config) as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)       
        
    # Set up a logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    
    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)
    
    # Stream output to file
    if "log" in hyper_config:
        savepath = hyper_config["log"]["save_path"] if "save_path" in hyper_config["log"] else "log.txt"
        mode = "a+" if bool(hyper_config["optuna"]["reload"]) else "w"
        fh = logging.FileHandler(savepath,
                                 mode=mode,
                                 encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        root.addHandler(fh)
        
    # Override other options in hyperparameter config file, if supplied.
    for name, val in args_dict.items():
        if val and (name in hyper_config):
            current_value = hyper_config["optuna"][name]
            logging.info(
                f"Overriding {name} in the hyperparameter configuration: {current_value} -> {val}"
            )
            hyper_config["optuna"][name] = val
        
    # Print the configurations to the logger
    logging.info("Current hyperparameter configuration settings:")
    for p, v in recursive_config_reader(hyper_config):
        full_path = ".".join([str(_p) for _p in p])
        logging.info(f"{full_path}: {v}")
    logging.info("Current model configuration settings:")
    for p, v in recursive_config_reader(model_config):
        full_path = ".".join([str(_p) for _p in p])
        logging.info(f"{full_path}: {v}")
        
    # Set up new db entry if reload = 0 
    reload_study = bool(hyper_config["optuna"]["reload"])
        
    # Check if save directory exists
    if not os.path.isdir(hyper_config["optuna"]["save_path"]):
        raise OSError(
            f'Create the save directory {hyper_config["optuna"]["save_path"]} and try again'
        )
        
    study_name = hyper_config["optuna"]["study_name"]
    storage = hyper_config["optuna"]["storage"]
    direction = hyper_config["optuna"]["direction"]
    single_objective = isinstance(direction, str)
    
    # Initialize the sampler
    if "sampler" not in hyper_config["optuna"]:
        if single_objective: # single-objective
            sampler = optuna.samplers.TPESampler()
        else: # multi-objective equivalent of TPESampler
            sampler = optuna.multi_objective.samplers.MOTPEMultiObjectiveSampler()
    else:
        sampler = samplers(hyper_config["optuna"]["sampler"])
        
    if "pruner" not in hyper_config["optuna"]:
        pruner = optuna.pruners.NopPruner()
    else:
        pruner = pruners(hyper_config["optuna"]["pruner"])
        
    if reload_study and not os.path.isfile(storage):
        logging.info(
            "No storage file exists yet, but the reload parameter was set to True. Overriding.")
        reload_study = False

    # Initiate a study for the first time
    if not reload_study:
        
        # Check the direction
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
        
        # Check if the study record already exists.
        try:
            optuna.load_study(
                study_name = study_name,
                storage = storage,
                sampler = sampler,
                pruner = pruner
            )
        except KeyError: # The study name was not in storage, can proceed
            pass
        
        except:
            if args_dict["override"]:
                message = f"Removing the study_name {study_name} that exists in storage {storage}."
                optuna.delete_study(
                    study_name = study_name,
                    storage = storage,
                    direction = direction,
                    sampler = sampler,
                    pruner = pruner
            )
            else:
                message = f"The study {study_name} already exists in storage and reload was False."
                message += f" Delete it from {storage}, and try again or rerun this script"
                message += f" with the flag: --override 1"
                raise OSError(message)
                
        # Create a new study in the storage object
        if single_objective:
            create_study = optuna.create_study(
                study_name = study_name,
                storage = storage,
                direction = direction,
                sampler = sampler,
                pruner = pruner
            )
        else:
            create_study = optuna.multi_objective.study.create_study(
                study_name = study_name,
                storage = storage,
                directions = direction,
                sampler = sampler,
                pruner = pruner
            )
            
    # Check to see if there are any broken trials        
    else:  
        logging.info(
            f"Checking the study for broken trials (those that did not complete 1 epoch before dying)"
        )
        if single_objective:
            study = optuna.load_study(
                study_name = study_name,
                storage = storage, 
                sampler = sampler,
                pruner = pruner,
            )
        else:
            study = optuna.multi_objective.study.load_study(
                study_name = study_name,
                storage = storage, 
                sampler = sampler,
                pruner = pruner,
            )
        study, removed = fix_broken_study(study, study_name, storage, direction, sampler, pruner)
        
        if len(removed):
            logging.info(
                f"Removing problematic trials {removed}."
            )
        else:
            logging.info("All trials check out!")
            
        
    # Override to create the database but skip submitting jobs. 
    create_db_only = True if args_dict["create_study"] else False
    
    # Stop here if arg is defined -- intention is that you manually run run.py for debugging purposes
    if create_db_only:
        logging.info(f"Created study {study_name} located at {storage}. Exiting.")
        sys.exit()
        
    ###############
    #
    # SLURM SUPPORT
    #
    ###############
                
    # Prepare launch script
    if "slurm" in hyper_config:
        launch_script = prepare_slurm_launch_script(hyper_config, model_config)
    
        # Save the configured script
        script_path = hyper_config["optuna"]["save_path"]
        script_location = os.path.join(script_path, "launch_slurm.sh")
        with open(script_location, "w") as fid:
            for line in launch_script:
                fid.write(f"{line}\n")

        # Launch the slurm jobs
        job_ids = []
        name_condition = "J" in hyper_config["slurm"]["batch"]
        slurm_job_name = hyper_config["slurm"]["batch"]["J"] if name_condition else "echo_trial"
        n_workers = hyper_config["slurm"]["jobs"]
        for worker in range(n_workers):
            w = subprocess.Popen(
                f"sbatch -J {slurm_job_name}_{worker} {script_location}",
                shell=True,
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE
            ).communicate()
            job_ids.append(
                w[0].decode("utf-8").strip("\n").split(" ")[-1]
            )
            logging.info(
                f"Submitted slurm batch job {worker + 1}/{n_workers} with id {job_ids[-1]}"
            )

        # Write the job ids to file for reference
        with open(os.path.join(script_path, "slurm_job_ids.txt"), "w") as fid:
            for line in job_ids:
                fid.write(f"{line}\n")
            
    ###############
    #
    # PBS SUPPORT
    #
    ###############
            
    if "pbs" in hyper_config:
        launch_script = prepare_pbs_launch_script(hyper_config, model_config)
        
        # Save the configured script
        script_path = hyper_config["optuna"]["save_path"]
        script_location = os.path.join(script_path, "launch_pbs.sh")
        with open(script_location, "w") as fid:
            for line in launch_script:
                fid.write(f"{line}\n")

        # Launch the slurm jobs
        job_ids = []
        name_condition = "N" in hyper_config["pbs"]["batch"]
        slurm_job_name = hyper_config["pbs"]["batch"]["N"] if name_condition else "echo_trial"
        n_workers = hyper_config["pbs"]["jobs"]
        for worker in range(n_workers):
            w = subprocess.Popen(
                f"qsub -N {slurm_job_name}_{worker} {script_location}",
                shell=True,
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE
            ).communicate()
            job_ids.append(
                w[0].decode("utf-8").strip("\n")
            )
            logging.info(
                f"Submitted pbs batch job {worker + 1}/{n_workers} with id {job_ids[-1]}"
            )

        # Write the job ids to file for reference
        with open(os.path.join(script_path, "pbs_job_ids.txt"), "w") as fid:
            for line in job_ids:
                fid.write(f"{line}\n")

                
if __name__ == "__main__":
    main()