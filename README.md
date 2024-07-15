# **E**arth **C**omputing **H**yperparameter **O**ptimization (ECHO): A distributed hyperparameter optimization package build with Optuna

### Install

To install a stable version of ECHO from PyPI, use the following command:
```bash
pip install echo-opt
```

Install the latest version of ECHO directly from github with the following command:
```bash
pip install git+https://github.com/NCAR/echo-opt.git
```


Several commands will be placed onto the PATH:
`echo-opt, echo-report, echo-run`

### Usage
Launch a new optimization study:

```bash
echo-opt hyperparameters.yml model_config.yml
```
Produce a report about the results saved in the study:

```bash
echo-report hyperparameters.yml [-p plot_config.yml] [-m model_config.yml]
```
Run one trial:

```bash
echo-run hyperparameters.yml model_config.yml
```

### Dependencies

There are three files that must be supplied to use the optimize script:

* A custom objective class that trains your model and returns the metric to be optimized.

* A configuration file specifying the hyperparameter optimization settings.

* A model configuration file that contains the information needed to train your model (see examples in the holodec and gecko projects).

### Custom objective class

The custom **Objective** class (objective.py) must inherit a **BaseObjective** class (which lives in base_objective.py), and must contain a method named **train** that returns the value of the optimization metric (in a dictionary, see below). There are example objective scripts for both torch and Keras in the examples directory. Your custom Objective class will inherit all of the methods and attributes from the BaseObjective. The Objective's train does not depend on the machine learning library used! For example, a simple template has the following structure:

```python
from echo.src.base_objective import *
from echo.src.pruners import KerasPruningCallback

class Objective(BaseObjective):

    def __init__(self, config, metric = "val_loss"):

        # Initialize the base class
        BaseObjective.__init__(self, config, metric)

    def train(self, trial, conf):

        # Make any custom edits to the model conf before using it to train a model.
        conf = custom_updates(trial, conf)

        # ... (load data sets, build model, etc)

        callbacks = [KerasPruningCallback(trial, self.metric, interval = 1)]
        result = Model.fit(..., callbacks=callbacks)

        results_dictionary = {
            "val_loss": result["val_loss"],
            "loss": result["loss"],
            #...
            "val_accuracy": result["val_accuracy"]
        }
        return results_dictionary
```
You can have as many inputs to your custom Objective as needed, as long as those that are required to initialize the base class are included. The Objective class will call the train method from the inherited thunder **__call__** method, and will finish up by calling the inherited save method that writes the metric(s) details to disk. Note that, due to the inheritance of the one class on the other, you do not have to supply these two methods, as they are in pre-coded in the base class. You can customize them at your leisure using overriding methods in your custom Objective. Check out the scripts base_objective.py and run.py to see how things are structured and called.

As noted, the metric used to toggle the model's training performance must be in the results dictionary. Other metrics that the user may want to track will be saved to disk if they are included in the results dictionary (the keys of the dictionary are used to name the columns in a pandas dataframe). See the example above where several metrics are being returned.

Note that the first line in the train method states that any custom changes to the model configuration (conf) must be done here. If custom changes are required, the user may supply a method named **custom_updates** in addition to the Objective class (you may save both in the same script, or import the method from somewhere else in your custom Objective script). See also the section **Custom model configuration updates** below for an example. 

Finally, if using Keras, you need to include the (customized) KerasPruningCallback that will allow optuna to terminate unpromising trials. We do something similar when using torch -- see the examples directory.

### Hyperparameter optimizer configuration

There are several fields: log, slurm, pbs, optuna, and variable subfields within each field. The log field allows us to save a file for printing messages and warnings that are placed in areas throughout the package. The slurm/pbs fields allows the user to specify how many GPU nodes should be used, and supports any slurm setting. The optuna field allows the user to configure the optimization procedure, including specifying which parameters will be used, as well as the performance metric. For example, consider the configuration settings:

```yaml
log: True
save_path: "/glade/work/schreck/repos/echo-opt/echo/examples/torch/fmc"

pbs:
  jobs: 10
  kernel: "ncar_pylib /glade/work/schreck/py37"
  bash: ["module load ncarenv/1.3 gnu/8.3.0 openmpi/3.1.4 python/3.7.5 cuda/10.1"]
  batch:
    l: ["select=1:ncpus=8:ngpus=1:mem=128GB", "walltime=12:00:00"]
    A: "NAML0001"
    q: "casper"
    N: "echo_trial"
    o: "echo_trial.out"
    e: "echo_trial.err"
slurm:
  jobs: 15
  kernel: "ncar_pylib /glade/work/schreck/py37"
  bash: ["module load ncarenv/1.3 gnu/8.3.0 openmpi/3.1.4 python/3.7.5 cuda/10.1"]
  batch:
    account: "NAML0001"
    gres: "gpu:v100:1"
    mem: "128G"
    n: 8
    t: "12:00:00"
    J: "echo_trial"
    o: "echo_trial.out"
    e: "echo_trial.err"
optuna:
  storage: "mlp.db"
  study_name: "mlp"
  storage_type: "sqlite"
  objective: "examples/torch/objective.py"
  metric: "val_loss"
  direction: "minimize"
  n_trials: 500
  gpu: True
  sampler:
    type: "TPESampler"
    n_startup_trials: 30 
  parameters:
    num_dense:
      type: "int"
      settings:
        name: "num_dense"
        low: 0
        high: 10
    dropout:
      type: "float"
      settings:
        name: "dr"
        low: 0.0
        high: 0.5
    optimizer:learning_rate:
      type: "loguniform"
      settings:
        name: "lr"
        low: 0.0000001
        high: 0.01
    model:activation:
      type: "categorical"
      settings:
        name: "activation"
        choices: ["relu", "linear", "leaky", "elu", "prelu"]
  enqueue:
  - num_dense: 1
    dropout: 0.2
    learning_rate: 0.001
    activation: "relu"
```
The save_path field sets the location where all generated data will be saved.
* save_path: Directory path where data will be saved. 

The log field allows you to save the logging details to file to save_path; they will always be printed to stdout. If this field is removed, logging details will only be printed to stdout.
* log: boolean to save log.txt in save_path.

The subfields within "pbs" and slurm" should mostly be familiar to you. In this example there would be 10 jobs submitted to pbs queue and 15 jobs to the slurm queue. Most HPCs just use one or the other, so make sure to only speficy what your system supports. The kernel field is optional and can be any call(s) to activate a conda/python/ncar_pylib/etc environment. Additional snippets that you might need in your launch script can be added to the list in the "bash" field. For example, as in the example above, loading modules before training a model is required. Note that the bash options will be run in order, and before the kernel field. Remove or leave the kernel field blank if you do not need it.

The subfields within the "optuna" field have the following functionality:
* storage: sqlite or mysql destination.
* study_name: The name of the study.
* storage_type: Choose "sqlite" or "maria" if a MariaDB is setup. 
 * If "sqlite", the storage field will automatically be appended to the save_path field (e.g. sql:///{save_path}/mlp.db)
 * If "maria", specify the full path including username:password in the storage field (for example, mysql://user:pw@someserver.ucar.edu/optuna).
* objective: The path to the user-supplied objective class
* metric: The metric to be used to determine the model performance. 
* direction: Indicates which direction the metric must go to represent improvement (pick from maximimize or minimize)
* n_trials: The number of trials in the study.
* gpu: Set to true to obtain the GPUs and their IDs
* sampler
  + type: Choose how optuna will do parameter estimation. The default choice both here and in optuna is the [Tree-structured Parzen Estimator Approach](https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f), [e.g. TPESampler](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf). See the optuna documentation for the different options. For some samplers (e.g. GridSearch) additional fields may be included (e.g. search_space). 
* parameters
  + type: Option to select an optuna trial setting. See the [optuna Trial documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html?highlight=suggest#optuna.trial.Trial.suggest_uniform) for what is available. Currently, this package supports the available options from optuna: "categorical", "discrete_uniform", "float", "int", "loguniform", and "uniform".
  + settings: This dictionary field allows you to specify any settings that accompany the optuna trial type. In the example above, the named num_dense parameter is stated to be an integer with values ranging from 0 to 10. To see all the available options, consolt the [optuna Trial documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html?highlight=suggest#optuna.trial.Trial.suggest_uniform)
* enqueue: [Optional] Adding this option will allow the user to add trials with pre-defined values when the study is first initialized, that will be run in order according to their id. Each entry added must be structured as a dictionary with the paramater names exactly matching all the hyperparameter name field in the parameters field.
  
Lastly, the "log" field allows you to save the logging details to file; they will always be printed to stdout. If this field is removed, logging details will only be printed to stdout.

### Model configuration

The model configuration file can be what you had been using up to this point to train your model, in other words no changes are necessary. This package will take the suggested hyperparameters from an optuna trial and make changes to the model configuration on the fly. This can either be done automatically with this package, or the user may supply an additional method for making custom changes. For example, consider the (truncated) configuration for training a model to predict properties with a dataset containing images:

```yaml
model:
  image_channels: 1
  hidden_dims: [3, 94, 141, 471, 425, 1122]
  z_dim: 1277
  dense_hidden_dims: [1000]
  dense_dropouts: [0.0]
  tasks: ["x", "y", "z", "d", "binary"]
  activation: "relu"
optimizer:
  type: "lookahead-diffgrad"
  learning_rate: 0.000631
  weight_decay: 0.0
trainer:
  start_epoch: 0
  epochs: 1
  clip: 1.0
  alpha: 1.0
  beta: 0.1
  path_save: "test"
 ```
The model configuration will be automatically updated if and only if the name of the parameter specified in the hyperparameter configuration, optuna.parameters can be used as a nested lookup key in the model configuration file. For example, observe in the hyperparameter configuration file above that the named parameter **optimizer:learning_rate** contains a colon, that is downstream used to split the named parameter into multiple keys that allow us to, starting from the top of the nested tree in the model configuration file, work our way down until the relevant field is located and the trial-suggested value is substituted in. In this example, the split keys are ["optimizer", "learning_rate"]. 

This scheme will work in general as long as the named parameter in optuna.parameters uses : as the separator, and once split, the resulting list can be used to locate the relevant field in the model configuration.

Note that optuna has a limited range of trial parameters types; all but one them being numerical in one form or another. If you wanted to optimize the activation layer(s) in your neural network as in the example above, you could go about that by utilizing the "categorical" trial suggestor. For example, the following list of activation layer names could be specified: ["relu", "linear", "leaky", "elu", "prelu"].


### Custom model configuration updates

You may additionally supply rules for updating the model configuration file, by including a method named **custom_updates**, which will make the desired changes to the configuration file with optuna trail parameter guesses.

In the example configurations described above, the hyperparameter configuration contained an optuna.parameters field "num_dense," but this field is not present in the model configuration. There is however a "dense_hiddden_dims" field in the model configuration that contains a list of the layer sizes in the model, where the number of layers is the length of the list. In our example just one layer specified but we want to vary that number. 

To use the "num_dense" hyperparameter from the hyperparameter configuration file, we can create the following method:

```python
def custom_updates(trial, conf):

    # Get list of hyperparameters from the config
    hyperparameters = conf["optuna"]["parameters"]

    # Now update some via custom rules
    num_dense = trial.suggest_discrete_uniform(**hyperparameters["num_dense"])

    # Update the config based on optuna's suggestion
    conf["model"]["dense_hidden_dims"] = [1000 for k in range(num_dense)]        

    return conf 
```

The method should be called first thing in the custom Objective.train method (see the example Objective above). You may have noticed that the configuration (named conf) contains both hyperparameter and model fields. This package will copy the hyperparameter optuna field to the model configuration for convenience, so that we can reduce the total number of class and method dependencies (which helps me keep the code generalized). This occurs in the run.py script.

### Custom plot settings for echo-report

The script report.py will load the current study, identify the best trial in the study, and will compute the relative importantance of each parameter using both fanova and MDI (see [here](https://optuna.readthedocs.io/en/v1.3.0/reference/importance.html) for details). 

Additionally, the script will create two figures, an optimization history plot and an intermediate values plot. If your metric returns two values to be optimized, only the pareto front plot will be generated. See the [documentation](https://optuna.readthedocs.io/en/v1.3.0/reference/visualization.html) for details on the plots. 

Note that ECHO only supports the [matplotlib](https://optuna.readthedocs.io/en/latest/reference/visualization/matplotlib.html) generated plots from Optuna, for now. Optuna's default is to use plot.ly, however not all LTS Jupyter-lab environments support that backend.

The user may customize the plots to a degree, by additionally supplying a plot configuration yaml file (named plot_config.yml above, and called as an optional argument using the parser -p or --plot). Currently, the user may only adjust the rcParam backend variables (see [here](https://matplotlib.org/3.3.3/tutorials/introductory/customizing.html) for a comprehensive list) plus a limited set of other variables (see below), 

```yaml
optimization_history: 
    save_path: '/glade/work/schreck/repos/holodec-ml/scripts/schreck/decoder/results/opt_multi_particle'
    set_xlim: [0, 100]
    set_ylim: [3e4, 1e6]
    set_xscale: "log"
    set_yscale: "log"
    rcparams: 
        'backend': 'ps'
        'lines.markersize'  : 4
        'axes.labelsize': 10
        'legend.fontsize': 10
        'xtick.labelsize': 10
        'ytick.labelsize': 10
        'xtick.top': True
        'xtick.bottom': True
        'ytick.right': True
        'ytick.left': True
        'xtick.direction': 'in'
        'ytick.direction': 'in'
        'font.serif'    : 'Helvetica'
        'figure.dpi'       : 600
        'figure.autolayout': True
        'legend.numpoints'     : 1
        'legend.handlelength'  : 1.0
        'legend.columnspacing' : 1.0
```

For the other supported plots, simply add or change "optimization_history" to "intermediate_values", or if optimizing more than one metric, "pareto_front".
