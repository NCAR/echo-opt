import warnings
warnings.filterwarnings("ignore")

from aimlutils.echo.src.trial_suggest import trial_suggest_loader
from collections import defaultdict
import copy, os, sys, random
import pandas as pd 
import logging
import optuna


logger = logging.getLogger(__name__)


def recursive_update(nested_keys, dictionary, update):
    if isinstance(dictionary, dict) and len(nested_keys) > 1:
        recursive_update(nested_keys[1:], dictionary[nested_keys[0]], update)
    else:
        dictionary[nested_keys[0]] = update


class BaseObjective:
    
    def __init__(self, config, metric = "val_loss", device = "cpu"):
        
        self.config = config
        self.metric = metric
        self.device = f"cuda:{device}" if device != "cpu" else "cpu"
        
        self.results = defaultdict(list)
        save_path = config["optuna"]["save_path"]
        self.results_fn = os.path.join(save_path, f"hyper_opt_{random.randint(0, 1e5)}.csv")
        while os.path.isfile(self.results_fn):
            rand_index = random.randint(0, 1e5)
            self.results_fn = os.path.join(save_path, f"hyper_opt_{rand_index}.csv")
            
        logger.info(f"Initialized an objective to be optimized with metric {metric}")
        logger.info(f"Using device {device}")
        logger.info(f"Saving study/trial results to local file {self.results_fn}")
    
    def update_config(self, trial):
        
        logger.info(
            f"Attempting to automatically update the model configuration using optuna's suggested parameters"
        )
        
        # Make a copy the config that we can edit
        conf = copy.deepcopy(self.config)

        # Update the fields that can be matched automatically (through the name field)
        updated = []
        hyperparameters = conf["optuna"]["parameters"]
        for named_parameter, update in hyperparameters.items():
            if ":" in named_parameter:
                recursive_update(
                    named_parameter.split(":"), 
                    conf,
                    trial_suggest_loader(trial, update))
                updated.append(named_parameter)
            else:
                if named_parameter in conf:
                    conf[named_parameter] = trial_suggest_loader(trial, update)
                    updated.append(named_parameter)
                    
        logger.info(f"Those that got updated automatically: {updated}")
        return conf
        
#     #Deprecated as of writing of report.py script 

    def save(self, trial, results_dict):
        
        # Make sure the relevant metric was placed into the results dictionary
        single_objective = isinstance(self.metric, str)
        if single_objective:
            if self.metric not in results_dict:
                raise OSError(
                    "You must return the metric result to the hyperparameter optimizer"
                )
        else:
            for metric in self.metric:
                if metric not in results_dict:
                    raise OSError(
                        "You must return the metric result to the hyperparameter optimizer"
                    )
        
        # Save the hyperparameters used in the trial
        self.results["trial"].append(trial.number)
        for param, value in trial.params.items():
            self.results[param].append(value)
        
        # Save the metric and "other metrics"
        for metric, value in results_dict.items():
            self.results[metric].append(value)
            
        # Save pruning boolean
        self.results["pruned"] = int(trial.should_prune())
        #self.results["complete"] = int(trial.state == optuna.trial.TrialState.COMPLETE)
        
        # Save the df of results to disk
        pd.DataFrame.from_dict(self.results).to_csv(self.results_fn)
        
        logger.info(
            f"Saving trial {trial.number} results to local file {self.results_fn}"
        )
        
        if single_objective:
            return results_dict[self.metric]
        else:
            return [result[metric] for metric in self.metric]
    
    def __call__(self, trial):
        
        # Automatically update the config, when possible
        conf = self.update_config(trial)
        
        # Train the model
        logger.info(
            f"Beginning to train the model using the latest parameters from optuna"
        )
        
        result = self.train(trial, conf)
        
        return self.save(trial, result)
    
    def train(self, trial, conf):
        raise NotImplementedError