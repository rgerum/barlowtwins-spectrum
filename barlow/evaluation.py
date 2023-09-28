import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


def plot_pca(model, dataset):
    from sklearn.decomposition import PCA

    x = dataset.data
    x = x.view(x.size(0), -1).type(torch.float32)
    y = dataset.targets
    z = model.encoder(x)
    # z = (z - z.mean(0)) / z.std(0)  # NxD

    pca = PCA(2)
    z0 = pca.fit_transform(z.numpy(force=True))

    plt.subplot(122)
    for i in range(10):
        index = (y == i).numpy(force=True)
        (l,) = plt.plot(z0[index, 0], z0[index, 1], "o", alpha=0.1)
    # plt.savefig("test_mnist.png")

    plt.subplot(121)
    pca = PCA(2)
    z0 = pca.fit_transform(x.numpy(force=True))
    for i in range(10):
        index = (y == i).numpy(force=True)
        (l,) = plt.plot(z0[index, 0], z0[index, 1], "o", alpha=0.1)

    plt.savefig("test_mnist_v3_weightdecay0.1.png")
    plt.show()


def plot_tsne(model, dataset):
    from sklearn.manifold import TSNE

    x = dataset.data
    x = x.view(x.size(0), -1).type(torch.float32)
    y = dataset.targets
    z = model.encoder(x)
    z = (z - z.mean(0)) / z.std(0)  # NxD

    pca = TSNE(n_components=2)
    z0 = pca.fit_transform(z)

    for i in range(10):
        index = y == i
        (l,) = plt.plot(z0[index, 0], z0[index, 1], "o")
    l
    plt.savefig("test_mnist_v3_weightdecay0.1.png")
    plt.show()
    data[1]


def one_hot(y, nclass):
    return np.eye(nclass)[y]


def accuracy(y1, y2):
    return np.mean(np.argmax(y1, axis=1) == np.argmax(y2, axis=1))


def evaluate_ridge():
    from sklearn.linear_model import Ridge

    clf = Ridge(alpha=1.0)

    i = iter(train_loader)
    data = next(i)
    x0 = data[0]
    x0 = x0.view(x.size(0), -1)
    y0 = data[1].numpy(force=True)
    z0 = model.encoder(x0).numpy(force=True)

    data = next(i)
    x1 = data[0]
    x1 = x1.view(x.size(0), -1)
    y1 = data[1].numpy(force=True)
    z1 = model.encoder(x1).numpy(force=True)

    clf.fit(x0.numpy(force=True), one_hot(y0, 10))
    zz1 = clf.predict(x1.numpy(force=True))
    print(accuracy(zz1, one_hot(y0, 10)))

    clf.fit(z0, one_hot(y0, 10))
    zz1 = clf.predict(z1)
    print(accuracy(zz1, one_hot(y0, 10)))


def show_history():
    import pandas as pd

    history = pd.read_csv("lightning_logs/version_21/metrics.csv")
    plt.plot(history.epoch, history.train_loss, "o")
    plt.loglog()
    plt.show()


def get_category_directions(encoder, x, y, dataset=None):
    # x = dataset.data
    x = x.view(x.size(0), -1).type(torch.float32)
    # y = dataset.targets
    z = encoder(x)
    # z = (z - z.mean(0)) / z.std(0)  # NxD

    vectors = z  # [index]
    vectors = vectors / torch.linalg.norm(vectors, axis=-1)[..., None]
    y_hot = torch.nn.functional.one_hot(y)
    directions2 = torch.sum(vectors[:, None, :] * y_hot[:, :, None], dim=0) / torch.sum(
        y_hot[:, :, None], dim=0
    )
    return directions2


def test_categorization(
    model, dataset_test, directions2=None, data_for_directions=None
):
    x = dataset_test.data
    x = x.view(x.size(0), -1).type(torch.float32)
    y = dataset_test.targets
    z = model.encoder(x)

    if directions2 is None:
        directions2 = get_category_directions(model, data_for_directions)

    comparison = torch.nn.CosineSimilarity(dim=2)(z[:, None], directions2[None, :])
    y_pred = comparison.argmax(1)
    return torch.sum(y == y_pred) / y.shape[0]


def test_categorization(y_test, z_test, directions2=None):
    comparison = torch.nn.CosineSimilarity(dim=2)(z_test[:, None], directions2[None, :])
    y_pred = comparison.argmax(1)
    return torch.sum(y_test == y_pred) / y_test.shape[0]


def get_category_directions(y_train, z_train):
    vectors = z_train
    vectors = vectors / torch.linalg.norm(vectors, axis=-1)[..., None]
    y_hot = torch.nn.functional.one_hot(y_train)
    directions2 = torch.sum(vectors[:, None, :] * y_hot[:, :, None], dim=0) / torch.sum(
        y_hot[:, :, None], dim=0
    )
    return directions2


def flatten(x):
    return x.view(x.size(0), -1).type(torch.float32)
