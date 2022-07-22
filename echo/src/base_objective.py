from echo.src.trial_suggest import trial_suggest_loader
from echo.src.config import recursive_config_reader, recursive_update
from collections import defaultdict
import copy
import glob
import os
import pandas as pd
import logging
import warnings

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


class BaseObjective:
    def __init__(self, config, metric="val_loss", device="cpu"):

        self.config = config
        self.metric = metric
        self._summon = False

    def set_properties(self, node_id=None, device="cpu"):
        if isinstance(device, list):
            self.device = [f"cuda:{d}" for d in device]
        else:
            self.device = f"cuda:{device}" if device != "cpu" else "cpu"
        self._summon = True
        self.worker_index = node_id
        self.results = defaultdict(list)
        if node_id is not None:
            save_path = os.path.join(
                self.config["optuna"]["save_path"], "trial_results"
            )
            os.makedirs(save_path, exist_ok=True)
            self.save_path = save_path
            self.results_fn = os.path.join(
                self.save_path, f"trial_results_{str(node_id)}.csv"
            )
        else:
            self.save_path = self.config["optuna"]["save_path"]
            self.results_fn = os.path.join(self.save_path, "trial_results.csv")

        node_id = 0 if node_id is None else node_id
        logger.info(f"Worker {node_id} is summoned.")
        logger.info(
            f"\tinitializing an objective to be optimized with metric {self.metric}"
        )
        logger.info(f"\tusing device(s) {self.device}")
        logger.info(f"\tsaving study/trial results to local file {self.results_fn}")

    def update_config(self, trial):

        logger.info(
            "Attempting to automatically update the model configuration using optuna's suggested parameters"
        )

        """ Make a copy the config that we can edit """
        conf = copy.deepcopy(self.config)

        """ Update the fields that can be matched automatically (through the name field) """
        updated = []
        hyperparameters = conf["optuna"]["parameters"]
        for named_parameter, update in hyperparameters.items():
            if ":" in named_parameter:
                recursive_update(
                    named_parameter.split(":"),
                    conf,
                    trial_suggest_loader(trial, update),
                )
                updated.append(named_parameter)
            else:
                if named_parameter in conf:
                    conf[named_parameter] = trial_suggest_loader(trial, update)
                    updated.append(named_parameter)

        observed = []
        for (k, v) in recursive_config_reader(conf):
            for u in updated:
                if ":".join(k) == u:
                    logger.info(f"\t{u} : {v}")
                    observed.append(":".join(k))
                elif k == u:
                    logger.info(f"\t{u} : {v}")
                    observed.append(k)

        not_updated = list(set(hyperparameters.keys()) - set(observed))
        for p in not_updated:
            logger.warning(f"\t{p} was not auto-updated by ECHO")
        if len(not_updated):
            logger.warning("Not all parameters were updated by ECHO")
            logger.warning(
                "There may be a mismatch between the model and hyper config files"
            )
            logger.warning("If using custom_updates, ignore this message")

        return conf

    def save(self, trial, results_dict):

        """Make sure the relevant metric was placed into the results dictionary"""
        single_objective = isinstance(self.metric, str)
        if single_objective:
            assert (
                self.metric in results_dict
            ), f"You must return the metric {self.metric} result to the hyperparameter optimizer"
        else:
            for metric in self.metric:
                assert (
                    metric in results_dict
                ), f"You must return the metric {metric} result to the hyperparameter optimizer"

        """ Save the hyperparameters used in the trial """
        self.results["trial"].append(trial.number)
        for param, value in trial.params.items():
            self.results[param].append(value)

        """ Save the metrics """
        for metric, value in results_dict.items():
            self.results[metric].append(value)

        """ Save pruning boolean """
        if isinstance(self.config["optuna"]["metric"], str):
            self.results["pruned"] = int(trial.should_prune())

        df = pd.DataFrame.from_dict(self.results)

        """ Save the df of results to disk """
        if os.path.isfile(self.results_fn):
            df = pd.concat(
                [df, pd.read_csv(self.results_fn, usecols=list(df.columns))]
            ).reset_index(drop=True)
        df = df.drop_duplicates(["trial"])
        df = df.sort_values(["trial"])
        df.to_csv(self.results_fn)

        logger.info(
            f"Saving trial {trial.number} results to local file {self.results_fn}"
        )

        """ If using multiple workers, merge all results so far"""
        tmp_results = glob.glob(f"{self.save_path}/trial_results*.csv")
        if len(tmp_results) > 1:
            tmp_save = os.path.join(
                self.config["optuna"]["save_path"], "trial_results.csv"
            )
            logger.info(f"Merging trial results and saving to {tmp_save}")
            df_full = pd.concat([pd.read_csv(x) for x in tmp_results])
            df_full.sort_values("trial").to_csv(tmp_save)

        if single_objective:
            return results_dict[self.metric]
        else:
            return [results_dict[metric] for metric in self.metric]

    def __call__(self, trial):

        """Secondary set-up of node_id and devices"""
        if not self._summon:
            self.set_properties()

        """ Automatically update the config, when possible """
        conf = self.update_config(trial)

        """ Train the model """
        logger.info(f"Beginning trial {trial.number}")
        result = self.train(trial, conf)

        """ Return the results """
        return self.save(trial, result)

    def train(self, trial, conf):
        raise NotImplementedError
