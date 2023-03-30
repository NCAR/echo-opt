import numpy as np
import torch
import random
import logging
import os
try:
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms
except ImportError as err:
    print("torch and torchvision are not installed. Please install both for this example to work.")
    raise err
from collections import defaultdict
from echo.src.base_objective import BaseObjective

import warnings

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


is_cuda = torch.cuda.is_available()
device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


class Net(nn.Module):
    def __init__(self, filter1, filter2, dropout, output_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, filter1, 3)
        self.conv2 = nn.Conv2d(filter1, filter2, 3)
        self.fc1 = nn.Linear(4 * filter2 * 3 * 3, output_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.dr1 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dr1(x)
        # print(x.shape)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=-1)
        return x


class Objective(BaseObjective):
    def __init__(self, config, metric="val_accuracy"):

        # Initialize the base class
        BaseObjective.__init__(self, config, metric)

    def train(self, trial, conf):
        filter1 = conf["filter1"]
        filter2 = conf["filter2"]
        learning_rate = conf["learning_rate"]
        batch_size = conf["batch_size"]
        dropout = conf["dropout"]
        epochs = conf["epochs"]
        seed = conf["seed"]
        num_classes = 10

        stopping_patience = conf["early_stopping_patience"]
        lr_patience = conf["lr_patience"]
        verbose = 0

        # Fix seed for reproducibility
        seed_everything(seed)

        # Load the data and split it between train and valid sets
        # (x_train, y_train), (x_valid, y_valid) = tf.keras.datasets.cifar10.load_data()

        # Scale images to the [0, 1] range
        # x_train = x_train.astype("float32") / 255
        # x_valid = x_valid.astype("float32") / 255

        # Resize images for pytorch
        # x_train = x_train.transpose((0, 3, 1, 2))
        # x_valid = x_valid.transpose((0, 3, 1, 2))

        # Wrap torch Dataset and Loader objects around the numpy arrays
        #trainset = torch.utils.data.TensorDataset(
        #    torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long()
        #)
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        validset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        #validset = torch.utils.data.TensorDataset(
        #    torch.from_numpy(x_valid).float(), torch.from_numpy(y_valid).long()
        #)

        valid_loader = torch.utils.data.DataLoader(
            validset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        # Load the model
        model = Net(filter1, filter2, dropout, num_classes).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

        # Load loss
        criterion = torch.nn.NLLLoss()

        # Load schedulers
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=lr_patience, verbose=verbose, min_lr=1.0e-13
        )

        results_dict = defaultdict(list)
        for epoch in range(epochs):  # loop over the dataset multiple times

            """Train"""
            train_loss, train_accuracy = [], []
            model.train()
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels.squeeze(-1))
                loss.backward()
                optimizer.step()

                # print statistics
                train_loss.append(loss.item())
                train_accuracy.append(
                    (torch.argmax(outputs, -1) == labels.squeeze(-1))
                    .float()
                    .mean()
                    .item()
                )

            """ Validate """
            val_loss, val_accuracy = [], []
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(valid_loader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # forward
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.squeeze(-1))

                    # print statistics
                    val_loss.append(loss.item())
                    val_accuracy.append(
                        (torch.argmax(outputs, -1) == labels.squeeze(-1))
                        .float()
                        .mean()
                        .item()
                    )

            results_dict["train_loss"].append(np.mean(train_loss))
            results_dict["train_accuracy"].append(np.mean(train_accuracy))
            results_dict["valid_loss"].append(np.mean(val_loss))
            results_dict["valid_accuracy"].append(np.mean(val_accuracy))

            if verbose:
                print_str = f"Epoch: {epoch + 1}"
                print_str += f' train_acc: {results_dict["train_accuracy"][-1]:.4f}'
                print_str += f' valid_acc: {results_dict["valid_accuracy"][-1]:.4f}'
                print(print_str)

            # Anneal learning rate
            lr_scheduler.step(1 - results_dict["valid_accuracy"][-1])

            # Early stopping
            best_epoch = [
                i
                for i, j in enumerate(results_dict["valid_accuracy"])
                if j == max(results_dict["valid_accuracy"])
            ][0]
            offset = epoch - best_epoch
            if offset >= stopping_patience:
                break

        return {key: value[best_epoch] for key, value in results_dict.items()}
