from echo.src.config import (
    config_check,
    configure_storage,
    configure_sampler,
    configure_pruner,
    recursive_config_reader,
    recursive_update,
)
from echo.src.trial_suggest import trial_suggest_loader
from echo.src.base_objective import BaseObjective
from echo.src.reporting import successful_trials, get_sec
from echo.src.pruners import KerasPruningCallback
from echo.optimize import (
    prepare_pbs_launch_script,
    prepare_slurm_launch_script,
    generate_batch_commands,
)
# import tensorflow as tf
import optuna
import pytest
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


# ── Helpers shared across new tests ──────────────────────────────────────────

def _minimal_hyper_config(direction="minimize"):
    """Return a minimal valid hyper_config for unit tests."""
    return {
        "save_path": "/tmp/echo_test",
        "optuna": {
            "study_name": "test",
            "storage": "test.db",
            "storage_type": "sqlite",
            "objective": __file__,  # any real file works
            "direction": direction,
            "metric": "val_loss",
            "n_trials": 10,
            "gpu": False,
        },
    }


def _make_trial(direction="minimize"):
    """Return a fresh in-memory Optuna Trial for testing suggest methods."""
    study = optuna.create_study(direction=direction)
    return study.ask()


# ── config_check validation tests ────────────────────────────────────────────

def test_config_check_missing_save_path():
    """config_check must raise when save_path is absent."""
    with pytest.raises(AssertionError, match="save_path"):
        config_check({"optuna": {}}, {})


def test_config_check_invalid_direction():
    """config_check must raise for an unsupported direction string."""
    cfg = _minimal_hyper_config()
    cfg["optuna"]["direction"] = "sideways"
    with pytest.raises(AssertionError):
        config_check(cfg, {})


def test_config_check_invalid_storage_type():
    """config_check must raise for an unsupported storage_type."""
    cfg = _minimal_hyper_config()
    cfg["optuna"]["storage_type"] = "postgres"
    with pytest.raises(AssertionError):
        config_check(cfg, {})


def test_config_check_multi_objective_valid():
    """config_check should pass for a valid multi-objective config."""
    cfg = _minimal_hyper_config()
    cfg["optuna"]["direction"] = ["minimize", "maximize"]
    assert config_check(cfg, {}) is True


def test_config_check_multi_objective_invalid_direction():
    """config_check must raise when any direction in the list is invalid."""
    cfg = _minimal_hyper_config()
    cfg["optuna"]["direction"] = ["minimize", "diagonal"]
    with pytest.raises(AssertionError):
        config_check(cfg, {})


def test_config_check_pbs_missing_walltime():
    """config_check must raise when the PBS batch config has no walltime."""
    cfg = _minimal_hyper_config()
    cfg["pbs"] = {"batch": {"l": ["select=1:ncpus=8"]}}
    with pytest.raises(AssertionError):
        config_check(cfg, {})


def test_config_check_slurm_missing_walltime():
    """config_check must raise when the SLURM batch config has no 't' key."""
    cfg = _minimal_hyper_config()
    cfg["slurm"] = {"batch": {}}
    with pytest.raises(AssertionError):
        config_check(cfg, {})


# ── configure_storage tests ───────────────────────────────────────────────────

def test_configure_storage_sqlite():
    """configure_storage with sqlite should return a valid sqlite:/// URL."""
    cfg = _minimal_hyper_config()
    storage = configure_storage(cfg)
    assert isinstance(storage, str)
    assert storage.startswith("sqlite:///")
    assert "test.db" in storage


def test_configure_storage_maria():
    """configure_storage with maria should return the raw storage string."""
    cfg = _minimal_hyper_config()
    cfg["optuna"]["storage_type"] = "maria"
    cfg["optuna"]["storage"] = "mysql://user:pw@server/optuna"
    storage = configure_storage(cfg)
    assert storage == "mysql://user:pw@server/optuna"


def test_configure_storage_nfs(tmp_path):
    """configure_storage with nfs should return a JournalStorage object."""
    cfg = _minimal_hyper_config()
    cfg["save_path"] = str(tmp_path)
    cfg["optuna"]["storage_type"] = "nfs"
    storage = configure_storage(cfg)
    # JournalStorage is not a plain string
    assert not isinstance(storage, str)


# ── configure_sampler tests ───────────────────────────────────────────────────

def test_configure_sampler_default_single_objective():
    """No sampler key → TPESampler for single-objective."""
    cfg = _minimal_hyper_config()
    sampler = configure_sampler(cfg)
    assert isinstance(sampler, optuna.samplers.TPESampler)


def test_configure_sampler_default_multi_objective():
    """No sampler key → TPESampler for multi-objective."""
    cfg = _minimal_hyper_config()
    cfg["optuna"]["direction"] = ["minimize", "maximize"]
    sampler = configure_sampler(cfg)
    assert isinstance(sampler, optuna.samplers.TPESampler)


def test_configure_sampler_random():
    """Explicit RandomSampler is instantiated correctly."""
    cfg = _minimal_hyper_config()
    cfg["optuna"]["sampler"] = {"type": "RandomSampler"}
    sampler = configure_sampler(cfg)
    assert isinstance(sampler, optuna.samplers.RandomSampler)


def test_configure_sampler_grid_requires_search_space():
    """GridSampler without search_space should raise OSError."""
    cfg = _minimal_hyper_config()
    cfg["optuna"]["sampler"] = {"type": "GridSampler"}
    with pytest.raises(OSError, match="search_space"):
        configure_sampler(cfg)


# ── configure_pruner tests ────────────────────────────────────────────────────

def test_configure_pruner_default():
    """No pruner key → NopPruner."""
    cfg = _minimal_hyper_config()
    pruner = configure_pruner(cfg)
    assert isinstance(pruner, optuna.pruners.NopPruner)


def test_configure_pruner_median():
    """Explicit MedianPruner is instantiated correctly."""
    cfg = _minimal_hyper_config()
    cfg["optuna"]["pruner"] = {"type": "MedianPruner"}
    pruner_obj = configure_pruner(cfg)
    assert isinstance(pruner_obj, optuna.pruners.MedianPruner)


# ── trial_suggest tests ───────────────────────────────────────────────────────

def test_trial_suggest_float():
    trial = _make_trial()
    result = trial_suggest_loader(
        trial, {"type": "float", "settings": {"name": "lr", "low": 0.0, "high": 1.0}}
    )
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_trial_suggest_int():
    trial = _make_trial()
    result = trial_suggest_loader(
        trial, {"type": "int", "settings": {"name": "layers", "low": 1, "high": 10}}
    )
    assert isinstance(result, int)
    assert 1 <= result <= 10


def test_trial_suggest_categorical():
    trial = _make_trial()
    result = trial_suggest_loader(
        trial,
        {"type": "categorical", "settings": {"name": "act", "choices": ["relu", "elu"]}},
    )
    assert result in ["relu", "elu"]


def test_trial_suggest_loguniform():
    """Legacy 'loguniform' type maps to suggest_float(log=True)."""
    trial = _make_trial()
    result = trial_suggest_loader(
        trial,
        {"type": "loguniform", "settings": {"name": "lr", "low": 1e-5, "high": 1e-1}},
    )
    assert isinstance(result, float)
    assert 1e-5 <= result <= 1e-1


def test_trial_suggest_uniform():
    """Legacy 'uniform' type maps to suggest_float."""
    trial = _make_trial()
    result = trial_suggest_loader(
        trial,
        {"type": "uniform", "settings": {"name": "dr", "low": 0.0, "high": 0.5}},
    )
    assert isinstance(result, float)
    assert 0.0 <= result <= 0.5


def test_trial_suggest_discrete_uniform():
    """Legacy 'discrete_uniform' with q maps to suggest_float(step=q)."""
    trial = _make_trial()
    result = trial_suggest_loader(
        trial,
        {
            "type": "discrete_uniform",
            "settings": {"name": "ddr", "low": 0.0, "high": 1.0, "q": 0.1},
        },
    )
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_trial_suggest_invalid_type():
    """Unknown type should raise AssertionError."""
    trial = _make_trial()
    with pytest.raises(AssertionError, match="not valid"):
        trial_suggest_loader(trial, {"type": "unknown_type", "settings": {}})


# ── recursive_config_reader / recursive_update tests ─────────────────────────

def test_recursive_config_reader_flat():
    """Flat dict yields single-element key paths."""
    d = {"a": 1, "b": 2}
    result = list(recursive_config_reader(d))
    assert (["a"], 1) in result
    assert (["b"], 2) in result


def test_recursive_config_reader_nested():
    """Nested dict yields full key paths to leaves."""
    d = {"optimizer": {"learning_rate": 0.001}}
    result = list(recursive_config_reader(d))
    assert (["optimizer", "learning_rate"], 0.001) in result


def test_recursive_update():
    """recursive_update modifies the correct nested field in-place."""
    d = {"optimizer": {"learning_rate": 0.001, "weight_decay": 0.0}}
    recursive_update(["optimizer", "learning_rate"], d, 0.01)
    assert d["optimizer"]["learning_rate"] == 0.01
    assert d["optimizer"]["weight_decay"] == 0.0  # unchanged


# ── reporting utility tests ───────────────────────────────────────────────────

def test_get_sec_hours():
    assert get_sec("01:00:00") == 3600


def test_get_sec_minutes():
    assert get_sec("00:30:00") == 1800


def test_get_sec_mixed():
    assert get_sec("12:30:45") == 12 * 3600 + 30 * 60 + 45


def test_successful_trials_all_complete():
    """successful_trials counts all COMPLETE trials."""
    study = optuna.create_study()
    study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=3)
    assert successful_trials(study) == 3


def test_successful_trials_includes_pruned():
    """successful_trials counts PRUNED trials as successful."""
    study = optuna.create_study()

    def objective(trial):
        trial.report(999.0, step=0)
        raise optuna.TrialPruned()

    study.optimize(objective, n_trials=2)
    assert successful_trials(study) == 2


# ── BaseObjective tests ───────────────────────────────────────────────────────

def _make_obj_config(save_path, metric="val_loss", parameters=None):
    if parameters is None:
        parameters = {}
    return {
        "optuna": {
            "save_path": str(save_path),
            "metric": metric,
            "parameters": parameters,
        }
    }


def test_base_objective_update_config_flat_key(tmp_path):
    """update_config substitutes a flat (non-nested) parameter."""
    config = _make_obj_config(tmp_path, parameters={
        "learning_rate": {
            "type": "float",
            "settings": {"name": "lr", "low": 1e-4, "high": 1e-2},
        }
    })
    config["learning_rate"] = 0.001  # pre-existing value in config

    study = optuna.create_study()
    trial = study.ask()
    obj = BaseObjective(config, metric="val_loss")
    obj.set_properties(node_id=None, device="cpu")
    updated = obj.update_config(trial)
    assert 1e-4 <= updated["learning_rate"] <= 1e-2


def test_base_objective_update_config_nested_key(tmp_path):
    """update_config handles colon-separated nested keys."""
    config = _make_obj_config(tmp_path, parameters={
        "optimizer:learning_rate": {
            "type": "float",
            "settings": {"name": "lr", "low": 1e-4, "high": 1e-2},
        }
    })
    config["optimizer"] = {"learning_rate": 0.001}

    study = optuna.create_study()
    trial = study.ask()
    obj = BaseObjective(config, metric="val_loss")
    obj.set_properties(node_id=None, device="cpu")
    updated = obj.update_config(trial)
    assert 1e-4 <= updated["optimizer"]["learning_rate"] <= 1e-2


def test_base_objective_save_single_objective(tmp_path):
    """save() writes a CSV and returns the scalar metric value."""
    config = _make_obj_config(tmp_path)
    study = optuna.create_study(direction="minimize")
    trial = study.ask()

    obj = BaseObjective(config, metric="val_loss")
    obj.set_properties(node_id=None, device="cpu")
    result = obj.save(trial, {"val_loss": 0.42, "loss": 0.5})

    assert result == 0.42
    csv_path = tmp_path / "trial_results.csv"
    assert csv_path.exists()
    import pandas as pd
    df = pd.read_csv(csv_path)
    assert "val_loss" in df.columns
    assert "loss" in df.columns


def test_base_objective_save_multi_objective(tmp_path):
    """save() with a list metric returns a list of values."""
    config = _make_obj_config(tmp_path, metric=["val_loss", "val_accuracy"])
    config["optuna"]["metric"] = ["val_loss", "val_accuracy"]
    study = optuna.create_study(directions=["minimize", "maximize"])
    trial = study.ask()

    obj = BaseObjective(config, metric=["val_loss", "val_accuracy"])
    obj.set_properties(node_id=None, device="cpu")
    result = obj.save(trial, {"val_loss": 0.3, "val_accuracy": 0.9})

    assert result == [0.3, 0.9]


def test_base_objective_save_missing_metric_raises(tmp_path):
    """save() raises AssertionError when the metric key is missing from results."""
    config = _make_obj_config(tmp_path)
    study = optuna.create_study(direction="minimize")
    trial = study.ask()

    obj = BaseObjective(config, metric="val_loss")
    obj.set_properties(node_id=None, device="cpu")
    with pytest.raises(AssertionError, match="val_loss"):
        obj.save(trial, {"loss": 0.5})


# ── KerasPruningCallback tests ────────────────────────────────────────────────

def test_keras_pruning_callback_triggers_prune():
    """on_epoch_end raises TrialPruned when the pruner decides to stop the trial."""
    study = optuna.create_study(pruner=optuna.pruners.ThresholdPruner(upper=1.0))
    trial = study.ask()
    callback = KerasPruningCallback(trial, monitor="val_loss")
    with pytest.raises(optuna.TrialPruned):
        # 999.0 exceeds upper=1.0 → should prune
        callback.on_epoch_end(epoch=0, logs={"val_loss": 999.0})


def test_keras_pruning_callback_no_prune():
    """on_epoch_end does not raise when the NopPruner is in use."""
    study = optuna.create_study()  # NopPruner by default
    trial = study.ask()
    callback = KerasPruningCallback(trial, monitor="val_loss")
    # Should complete without raising
    callback.on_epoch_end(epoch=0, logs={"val_loss": 0.5})


def test_keras_pruning_callback_missing_monitor_key():
    """on_epoch_end is a no-op when the monitored metric is absent from logs."""
    study = optuna.create_study()
    trial = study.ask()
    callback = KerasPruningCallback(trial, monitor="val_loss")
    # 'val_loss' not in logs — should return without error or pruning
    callback.on_epoch_end(epoch=0, logs={"loss": 0.5})


# ── generate_batch_commands tests ─────────────────────────────────────────────

def _batch_cfg(batch_type, gpus_per_node=None, tasks_per_worker=None):
    """Minimal config dict for generate_batch_commands tests."""
    inner = {}
    if gpus_per_node is not None:
        inner["gpus_per_node"] = gpus_per_node
    if tasks_per_worker is not None:
        inner["tasks_per_worker"] = tasks_per_worker
    return {batch_type: inner}


def test_generate_batch_commands_single_worker():
    """No GPUs, no tasks_per_worker → single command, no & or wait."""
    cfg = _batch_cfg("pbs")
    cmds = generate_batch_commands(cfg, "pbs", "echo-run h.yml m.yml", "$PBS_JOBID")
    assert len(cmds) == 1
    assert "&" not in cmds[0]
    assert "wait" not in cmds[0]
    assert "echo-run" in cmds[0]


def test_generate_batch_commands_gpu_pinning():
    """With gpus_per_node=4, each GPU gets its own process with correct CUDA_VISIBLE_DEVICES."""
    cfg = _batch_cfg("pbs", gpus_per_node=4)
    cmds = generate_batch_commands(cfg, "pbs", "echo-run h.yml m.yml", "$PBS_JOBID")
    # Expect 4 worker lines + 1 wait
    worker_lines = [l for l in cmds if "echo-run" in l]
    assert len(worker_lines) == 4
    for i, line in enumerate(worker_lines):
        assert f"CUDA_VISIBLE_DEVICES={i}" in line
        assert line.endswith("&")
    assert cmds[-1] == "wait"


def test_generate_batch_commands_gpu_pinning_no_duplicate_devices():
    """Each GPU device ID appears exactly once across all worker lines."""
    cfg = _batch_cfg("pbs", gpus_per_node=4)
    cmds = generate_batch_commands(cfg, "pbs", "echo-run h.yml m.yml", "$PBS_JOBID")
    worker_lines = [l for l in cmds if "echo-run" in l]
    seen_devices = [int(l.split("CUDA_VISIBLE_DEVICES=")[1].split(" ")[0]) for l in worker_lines]
    assert sorted(seen_devices) == list(range(4))


def test_generate_batch_commands_tasks_per_worker_no_gpu():
    """tasks_per_worker without GPUs → N background processes, no CUDA prefix."""
    cfg = _batch_cfg("slurm", tasks_per_worker=3)
    cmds = generate_batch_commands(cfg, "slurm", "echo-run h.yml m.yml", "$SLURM_JOB_ID")
    worker_lines = [l for l in cmds if "echo-run" in l]
    assert len(worker_lines) == 3
    for line in worker_lines:
        assert "CUDA_VISIBLE_DEVICES" not in line
        assert line.endswith("&")
    assert cmds[-1] == "wait"


def test_generate_batch_commands_no_sleep_lines():
    """No sleep lines should appear in any generated script."""
    for cfg in [
        _batch_cfg("pbs", gpus_per_node=2),
        _batch_cfg("pbs", tasks_per_worker=2),
        _batch_cfg("pbs"),
    ]:
        cmds = generate_batch_commands(cfg, "pbs", "echo-run h.yml m.yml", "$PBS_JOBID")
        assert not any("sleep" in l for l in cmds), f"sleep found in: {cmds}"


def test_generate_batch_commands_independent_calls():
    """Each call to generate_batch_commands starts with a fresh list (no mutable default bleed)."""
    cfg = _batch_cfg("pbs")
    cmds1 = generate_batch_commands(cfg, "pbs", "echo-run h.yml m.yml", "$PBS_JOBID")
    cmds2 = generate_batch_commands(cfg, "pbs", "echo-run h.yml m.yml", "$PBS_JOBID")
    assert len(cmds1) == len(cmds2) == 1
