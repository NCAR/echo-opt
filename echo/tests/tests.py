import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")


def load_example_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    assert x_train.shape == (50000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_test.shape == (10000, 1)