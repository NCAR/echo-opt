from echo.src.config import config_check
from echo.optimize import prepare_pbs_launch_script, prepare_slurm_launch_script
# import tensorflow as tf
import warnings
import yaml
import os

warnings.filterwarnings("ignore")


#def test_load_example_data():
#    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
#    assert x_train.shape == (50000, 32, 32, 3)
#    assert y_train.shape == (50000, 1)
#    assert x_test.shape == (10000, 32, 32, 3)
#    assert y_test.shape == (10000, 1)


def test_read_keras_config():

    _hyper_config = "echo/examples/keras/hyperparameter.yml"
    _model_config = "echo/examples/keras/model_config.yml"

    assert os.path.isfile(_hyper_config)
    assert os.path.isfile(_model_config)

    with open(_hyper_config) as f:
        hyper_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(_model_config) as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)

    config_check(hyper_config, model_config)


def test_read_torch_config():

    _hyper_config = "echo/examples/torch/hyperparameter.yml"
    _model_config = "echo/examples/torch/model_config.yml"

    assert os.path.isfile(_hyper_config)
    assert os.path.isfile(_model_config)

    with open(_hyper_config) as f:
        hyper_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(_model_config) as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)

    config_check(hyper_config, model_config)


# ── PBS launch script tests ───────────────────────────────────────────────────

def _base_pbs_config(extra_select=None):
    """Return a minimal hyper_config dict with a PBS section."""
    select = "select=1:ncpus=8:ngpus=1:mem=64GB:gpu_type=a100_80gb"
    if extra_select:
        select = extra_select
    return {
        "save_path": "/tmp/echo_test",
        "pbs": {
            "jobs": 1,
            "batch": {
                "l": [select, "walltime=04:00:00"],
                "A": "TEST0001",
                "q": "casper",
                "N": "test_job",
                "o": "/tmp/echo_test/out",
                "e": "/tmp/echo_test/err",
            },
            "bash": ["source ~/.bashrc"],
            "kernel": "conda activate echo",
        },
        "optuna": {
            "study_name": "test",
            "storage": "test.db",
            "storage_type": "sqlite",
            "objective": __file__,  # just needs to be a real file
            "direction": "maximize",
            "metric": "val_accuracy",
            "n_trials": 10,
            "gpu": True,
        },
    }


def test_pbs_launch_script_has_shebang():
    hyper_config = _base_pbs_config()
    script = prepare_pbs_launch_script(hyper_config, {})
    assert script[0] == "#!/bin/bash -l", f"Expected shebang, got: {script[0]}"


def test_pbs_launch_script_walltime_present():
    hyper_config = _base_pbs_config()
    script = prepare_pbs_launch_script(hyper_config, {})
    assert any("walltime=04:00:00" in line for line in script), (
        "walltime not found in PBS script"
    )


def test_pbs_launch_script_gpu_type_in_select():
    """The gpu_type parameter must appear in the #PBS -l select line."""
    hyper_config = _base_pbs_config()
    script = prepare_pbs_launch_script(hyper_config, {})
    select_lines = [l for l in script if "#PBS -l" in l and "select=" in l]
    assert len(select_lines) >= 1, "No #PBS -l select line found"
    assert any("gpu_type=a100_80gb" in l for l in select_lines), (
        f"gpu_type=a100_80gb not found in select lines: {select_lines}"
    )


def test_pbs_launch_script_various_gpu_types():
    """All valid Casper gpu_type values should pass through unchanged."""
    valid_gpu_types = [
        "v100_32gb", "a100_40gb", "a100_80gb", "h100_80gb",
        "mi300a", "gp100_16gb", "l40_45gb",
    ]
    for gpu_type in valid_gpu_types:
        select = f"select=1:ncpus=8:ngpus=1:mem=64GB:gpu_type={gpu_type}"
        hyper_config = _base_pbs_config(extra_select=select)
        script = prepare_pbs_launch_script(hyper_config, {})
        select_lines = [l for l in script if "#PBS -l" in l and "select=" in l]
        assert any(gpu_type in l for l in select_lines), (
            f"gpu_type={gpu_type} not found in PBS script lines: {select_lines}"
        )


def test_pbs_launch_script_echo_run_command():
    """The script must contain an echo-run invocation."""
    hyper_config = _base_pbs_config()
    script = prepare_pbs_launch_script(hyper_config, {})
    assert any("echo-run" in line for line in script), (
        "echo-run command not found in PBS script"
    )


def test_pbs_launch_script_bash_lines():
    hyper_config = _base_pbs_config()
    script = prepare_pbs_launch_script(hyper_config, {})
    assert any("source ~/.bashrc" in line for line in script)
    assert any("conda activate echo" in line for line in script)


# ── SLURM launch script tests ─────────────────────────────────────────────────

def _base_slurm_config():
    return {
        "save_path": "/tmp/echo_test",
        "slurm": {
            "jobs": 1,
            "batch": {
                "J": "test_job",
                "t": "04:00:00",
                "N": "1",
                "n": "1",
                "mem": "64G",
                "partition": "gpu",
            },
            "bash": ["source ~/.bashrc"],
            "kernel": "conda activate echo",
        },
        "optuna": {
            "study_name": "test",
            "storage": "test.db",
            "storage_type": "sqlite",
            "objective": __file__,
            "direction": "maximize",
            "metric": "val_accuracy",
            "n_trials": 10,
            "gpu": True,
        },
    }


def test_slurm_launch_script_has_shebang():
    hyper_config = _base_slurm_config()
    script = prepare_slurm_launch_script(hyper_config, {})
    assert script[0] == "#!/bin/bash -l", f"Expected shebang, got: {script[0]}"


def test_slurm_launch_script_sbatch_options():
    hyper_config = _base_slurm_config()
    script = prepare_slurm_launch_script(hyper_config, {})
    sbatch_lines = [l for l in script if l.startswith("#SBATCH")]
    assert len(sbatch_lines) > 0, "No #SBATCH lines found"


def test_slurm_launch_script_echo_run_command():
    hyper_config = _base_slurm_config()
    script = prepare_slurm_launch_script(hyper_config, {})
    assert any("echo-run" in line for line in script), (
        "echo-run command not found in SLURM script"
    )


def test_slurm_launch_script_bash_lines():
    hyper_config = _base_slurm_config()
    script = prepare_slurm_launch_script(hyper_config, {})
    assert any("source ~/.bashrc" in line for line in script)
    assert any("conda activate echo" in line for line in script)
