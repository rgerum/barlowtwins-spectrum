import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from barlow_model import BarlowModel
from evaluation import plot_pca, test_categorization, flatten, get_category_directions

torch.set_float32_matmul_precision("medium")


class Trans(nn.Module):
    def forward(self, input):
        input = input.type(torch.float32)
        if len(input.shape) == 3:
            input = input[:, None]
        return input  # /255


# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(
    nn.Linear(28 * 28, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Linear(128, 32),
    # nn.ReLU(),
    # nn.Linear(64, 2),
)
encoder = nn.Sequential(
    Trans(),
    nn.Conv2d(1, 16, 3),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 16, 3),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(400, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
)

projector = nn.Sequential(
    nn.Linear(128, 256),
    # nn.BatchNorm1d(32),
    # nn.ReLU(),
    # nn.Linear(32, 10),
    # nn.ReLU(),
    # nn.Linear(64, 2),
)
# projector = lambda x: x

augmenter = nn.Sequential(
    # transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.GaussianBlur(3),
    transforms.ColorJitter(0.5),
)

# setup data
dataset = FashionMNIST(os.getcwd(), download=True, transform=ToTensor())

dataset0 = MNIST(os.getcwd(), download=True, transform=ToTensor())
# dataset0 = MNIST(os.getcwd(), download=True, transform=lambda x: ToTensor()(x)x[:, None])

dataset_test = MNIST(os.getcwd(), download=True, train=False, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset, batch_size=128)
test_loader = utils.data.DataLoader(dataset_test, batch_size=10000)


def transform(x):
    x = x.type(torch.float32)
    x = x  # /255
    return x


# init the autoencoder
model = BarlowModel(encoder, projector, augmenter, dataset0, 1)

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = pl.Trainer(max_epochs=100, accelerator="gpu")
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)

plot_pca(model, dataset)

# print(test_categorization(model, dataset_test))

if 0:
    x_train, y_train = flatten(dataset.data), dataset.targets
    x_test, y_test = flatten(dataset_test.data), dataset_test.targets
    z_train = encoder(x_train)
    z_test = encoder(x_test)

    directions = get_category_directions(y_train, z_train)
    test_categorization(y_test, z_test, directions)
