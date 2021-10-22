from typing import Dict
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import logging
import optuna
import yaml
import sys
import os
import warnings
warnings.filterwarnings("ignore")


def args():
    parser = ArgumentParser(description="report.py: Get the status/progress of a hyperparameter study"
                            )

    parser.add_argument("hyperparameter", type=str, help="Path to the hyperparameter configuration containing your inputs."
                        )

    parser.add_argument(
        "-p",
        "--plot",
        dest="plot",
        type=str,
        default=False,
        help="A yaml structured file containining settings for matplotlib/pylab objects"
    )

    parser.add_argument(
        "-t",
        "--n_trees",
        dest="n_trees",
        type=int,
        default=64,
        help="The number of trees to use in parameter importance models. Default is 64."
    )

    parser.add_argument(
        "-d",
        "--max_depth",
        dest="max_depth",
        type=int,
        default=64,
        help="The maximum depth to use in parameter importance models. Default is 64."
    )

    return vars(parser.parse_args())


def update_figure(fig: mpl.figure.Figure,
                  params: Dict[str, str] = False) -> mpl.figure.Figure:
    """
    Updates some mpl Figure parameters. Only limited support for now.
    In a future version the optuna plots will be moved here 
    and expanded customization will be enabled.

    Returns a matplotlib Figure

    Inputs: 
        fig: a matplotlib Figure
        params: a dictionary containing mpl fields
    """

    if params is False:
        fig.set_yscale("log")
        mpl.rcParams.update({"figure.dpi": 300})
    else:
        if "rcparams" in params:
            mpl.rcParams.update(**params["rcparams"])
        if "set_xlim" in params:
            fig.set_xlim(params["set_xlim"])
        if "set_ylim" in params:
            fig.set_ylim(params["set_ylim"])
        if "set_xscale" in params:
            fig.set_xscale(params["set_xscale"])
        if "set_yscale" in params:
            fig.set_yscale(params["set_yscale"])

    plt.tight_layout()
    return fig


def plot_wrapper(study: optuna.study.Study,
                 identifier: str,
                 save_path: str,
                 params: Dict[str, str] = False):
    """
    Creates and saves an intermediate values plot.

    Does not return. 

    Inputs: 
        study: an Optuna study object
        identifier: a string identifier for selecting the optuna plot method
        save_path: a path where the plot should be saved
        params: a dictionary containing mpl fields. Default = False
    """

    flag = isinstance(params, dict)
    if flag and identifier in params:
        params = params[identifier]
    else:
        flag = False

    # Use optunas mpl object for now
    if identifier == "intermediate_values":
        fig = optuna.visualization.matplotlib.plot_intermediate_values(study)
    elif identifier == "optimization_history":
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    elif identifier == "pareto_front":
        fig = optuna.multi_objective.visualization.plot_pareto_front(study)
    else:
        raise OSError(
            f"An incorrect optuna plot identifier {identifier} was used")

    fig = update_figure(fig, params)

    if flag and "save_path" in params:
        save_path = params["save_path"]

    figure_save_path = os.path.join(save_path, f"{identifier}.pdf")
    plt.savefig(figure_save_path)

    logging.info(
        f"Saving the {identifier} plot to file at {figure_save_path}"
    )


if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise OSError(
            "Usage: python report.py hyperparameter.yml [optional arguments]"
            "To see the available parser options: python report.py --help"
        )

    args_dict = args()

    hyper_config = args_dict.pop("hyperparameter")
    plot_config = args_dict.pop("plot") if "plot" in args_dict else False

    # Options for the parameter importance tree models
    n_trees = args_dict.pop("n_trees")
    max_depth = args_dict.pop("max_depth")

    # Check if hyperparameter config file exists
    if os.path.isfile(hyper_config):
        with open(hyper_config) as f:
            hyper_config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise OSError(
            f"Hyperparameter optimization config file {hyper_config} does not exist"
        )

    if plot_config is not False:
        if os.path.isfile(plot_config):
            with open(plot_config) as p:
                plot_config = yaml.load(p, Loader=yaml.FullLoader)
        else:
            raise OSError(
                f"Hyperparameter optimization plot file {plot_config} does not exist"
            )

    # Set up a logger
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    save_path = hyper_config["optuna"]["save_path"]
    study_name = hyper_config["optuna"]["study_name"]
    storage = hyper_config["optuna"]["storage"]
    reload_study = bool(hyper_config["optuna"]["reload"])
    cached_study = f"{save_path}/{study_name}"

    direction = hyper_config["optuna"]["direction"]
    single_objective = isinstance(direction, str)

    # Load from database
    #storage = f'postgresql+psycopg2://john:schreck@localhost/{cached_study}'
    #storage = f"sqlite:///{cached_study}"

    if single_objective:
        study = optuna.load_study(study_name=study_name, storage=storage)
    else:
        study = optuna.multi_objective.study.load_study(
            study_name=study_name,
            storage=storage
        )

    # Check a few other stats
    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    logging.info(
        f'Number of requested trials per worker: {hyper_config["optuna"]["n_trials"]}')
    logging.info(f"Number of trials in the database: {len(study.trials)}")
    logging.info(f"Number of pruned trials: {len(pruned_trials)}")
    logging.info(f"Number of completed trials: {len(complete_trials)}")

    if len(complete_trials) == 0:
        logging.info("There are no complete trials in this study.")
        logging.info(
            "Wait until the workers finish a few trials and try again.")
        sys.exit()

    logging.info(f"Best trial: {study.best_trial.value}")

    if len(complete_trials) > 1:
        try:
            f_importance = optuna.importance.FanovaImportanceEvaluator(
                n_trees=n_trees, max_depth=max_depth).evaluate(study=study)
            logging.info(f"fANOVA parameter importance {dict(f_importance)}")
            mdi_importance = optuna.importance.MeanDecreaseImpurityImportanceEvaluator(
                n_trees=n_trees, max_depth=max_depth).evaluate(study=study)
            logging.info(
                f"Mean decrease impurity (MDI) parameter importance {dict(mdi_importance)}")
        except Exception as E:  # Encountered zero total variance in all trees.
            logging.warning(
                f"Failed to compute parameter importance due to error: {E}")
            pass

    logging.info("Best parameters in the study:")
    for param, val in study.best_params.items():
        logging.info(f"{param}: {val}")

    if len(study.trials) < hyper_config["optuna"]["n_trials"]:
        logging.warning(
            "Not all of the trials completed due to the wall-time."
        )
        logging.warning(
            "Set reload = 1 in the hyperparameter config and resubmit some more workers to finish!"
        )

    save_fn = os.path.join(save_path, f"{study_name}.csv")
    logging.info(f"Saving the results of the study to file at {save_fn}")
    study.trials_dataframe().to_csv(save_fn, index=None)

    if single_objective:

        # Plot the optimization_history
        plot_wrapper(study, "optimization_history", save_path, plot_config)

        # Plot the intermediate_values
        plot_wrapper(study, "intermediate_values", save_path, plot_config)

    else:
        # Plot the pareto front
        plot_wrapper(study, "pareto_front", save_path, plot_config)
