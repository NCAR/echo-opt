import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.inspection import partial_dependence
import logging
import warnings

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


def partial_dep(fn, input_cols, output_col, verbose=0, model_type='rf'):

    df = fn[~fn[output_col].isna()].copy()

    hot = {}
    for param, dtype in df[input_cols].dtypes.to_dict().items():
        if not any(x in str(dtype) for x in ["int", "float", "bool"]):
            hot[param] = LabelEncoder()

    if len(hot):
        for param, le in hot.items():
            df[param] = le.fit_transform(df[param])

    objective = "reg:squarederror"
    criterion = "squared_error"
    learning_rate = 0.075
    n_estimators = 1000
    max_depth = 10
    n_jobs = 8
    colsample_bytree = 0.8995496645826047
    gamma = 0.6148001693726943
    learning_rate = 0.07773680788294579
    max_depth = 10
    subsample = 0.7898672617361431
    seed = 1000

    X = df[input_cols].copy()
    y = df[output_col].copy()
    x_train, _X, y_train, _y = train_test_split(X, y, test_size=0.2, random_state=seed)
    x_valid, x_test, y_valid, y_test = train_test_split(
        _X, _y, test_size=0.5, random_state=seed
    )

    #     xscaler = #StandardScaler()
    #     yscaler = StandardScaler()

    #     x_train = xscaler.fit_transform(x_train)
    #     x_valid = xscaler.transform(x_valid)
    #     x_test = xscaler.transform(x_test)

    #     y_train = yscaler.fit_transform(np.expand_dims(y_train, -1))
    #     y_valid = yscaler.transform(np.expand_dims(y_valid, -1))
    #     y_test = yscaler.transform(np.expand_dims(y_test, -1))

    if model_type=='xgb':
        xgb_model = xgb.XGBRegressor(
            objective=objective,
            random_state=seed,
            # gpu_id = device,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            subsample=subsample,
            n_jobs=n_jobs,
        )

        xgb_model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            early_stopping_rounds=10,
            verbose=verbose,
        )

        model = xgb_model

    if model_type=='rf':
        rf_model = RandomForestRegressor(
            criterion=criterion,
            random_state=seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            verbose=verbose,
            max_samples=subsample,
            n_jobs=n_jobs,
        )

        rf_model.fit(
            x_train,
            y_train,
        )

        model = rf_model

    train_sh = y_train.shape[0]
    valid_sh = y_valid.shape[0]
    test_sh = y_test.shape[0]

    info = f"\tTrain ({train_sh}) / valid ({valid_sh}) / test ({test_sh})\tR2 scores: "
    info += f"\t{model.score(x_train, y_train):.2f} / "
    info += f"{model.score(x_valid, y_valid):.2f} / "
    info += f"{model.score(x_valid, y_valid):.2f} "
    logger.info(info)

    return model, x_train, hot


def plot_partial_dependence(f, metrics, save_path, verbose=0, model_type='rf'):
    input_cols = [x for x in f.columns if x.startswith("params_")]
    if isinstance(metrics, list):
        output_cols = [f"values_{k}" for k in range(len(metrics))]
    else:
        output_cols = ["value"]
        metrics = [metrics]

    if len(input_cols) < 9:
        cols = 2
    elif len(input_cols) < 16:
        cols = 3
    else:
        cols = 4
    num = int(np.ceil(len(input_cols) / cols))
    features = range(len(input_cols))

    for p, metric in enumerate(metrics):

        logger.info(f"Fitting {model_type} model to predict {metric} partial dependence")
        model, X, hot = partial_dep(f, input_cols, output_cols[p],
                verbose=verbose, model_type=model_type)

        fig, ax = plt.subplots(
            cols, num, figsize=(10, 10 / 1.61), sharex=False, sharey=False, dpi=300
        )

        save_name = f"{save_path}/partial_dependence_{metric}.png"
        logger.info(f"\tPlotting and saving partial dependences to {save_name}")

        outer = 0
        features = range(len(input_cols))
        for k, feature in enumerate(features):
            pd_result = partial_dependence(model, X, feature, grid_resolution=50)
            x = pd_result["average"]
            y = pd_result["values"]
            
            if input_cols[k] in hot:
                
                # if the dataset is too small, we cant yet make PD figures
                try:
                    y[0] = hot[input_cols[k]].inverse_transform(y[0])
                except ValueError:
                    continue
                        
            # x_rescaled = xscaler.inverse_transform(np.expand_dims(x[0], -1))
            ax[outer][k % num].plot(y[0], x[0], "b-")
            if any([type(x) == str for x in y[0]]):
                ax[outer][k % num].set_xticklabels(y[0], rotation=90)
            if input_cols[k].startswith("params_"):
                xlabel = "_".join(input_cols[k].split("_")[1:])
            else:
                xlabel = input_cols[k]
            ax[outer][k % num].set_xlabel(xlabel)
            if k % num == 0:
                ax[outer][k % num].set_ylabel(f"{metric} PD")
            if input_cols[k] not in hot:
                if max(y[0]) / min(y[0]) > 100:
                    ax[outer][k % num].set_xscale("log")
            if (k + 1) % num == 0:
                outer += 1

        plt.tight_layout()
        plt.savefig(save_name, bbox_inches="tight", dpi=300)

        del fig
