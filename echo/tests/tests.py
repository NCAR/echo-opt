from echo.src.config import config_check
import tensorflow as tf
import warnings
import yaml
import sys
import os 
warnings.filterwarnings("ignore")


def test_load_example_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    assert x_train.shape == (50000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_test.shape == (10000, 1)
    
    
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