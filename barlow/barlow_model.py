import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from torchvision import transforms
from barlow_loss import barlow_loss
from evaluation import test_categorization, get_category_directions, flatten
from spectral_reg import get_alpha, get_alpha_mse
import numpy as np


# define the LightningModule
class BarlowModel(pl.LightningModule):
    def __init__(self, encoder, projector, augmenter, data_for_directions, lambd=1):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.augment = augmenter
        self.lambd = lambd
        self.data_for_directions = data_for_directions

        self.data_for_directions_y = None
        self.data_for_directions_x = None

        index = []
        for i in range(10):
            index.append(
                torch.where(self.data_for_directions.targets == i)[0][:1000].numpy()
            )
        self.index = np.array(index).T

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x1 = x  # .view(x.size(0), -1)
        x2 = self.augment(x)  # .view(x.size(0), -1)
        z_a = self.encoder(x1)

        alpha, mse = get_alpha(z_a)
        mse = get_alpha_mse(z_a, 1)

        self.log("train_alpha", alpha)
        self.log("train_alpha_mse", mse)

        z_a = self.projector(z_a)
        z_b = self.projector(self.encoder(x2))

        loss = barlow_loss(z_a, z_b, self.lambd)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss + mse * 10

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-6, weight_decay=0.01)
        return optimizer

    def validation_accuracy(self, x, y, n=1):
        acc = []
        if self.data_for_directions_x is None:
            self.data_for_directions_y = torch.tensor(
                self.data_for_directions.targets
            ).to(y.device)
            self.data_for_directions_x = torch.tensor(self.data_for_directions.data).to(
                y.device
            )
        for i in range(10):
            examples = self.index[i : i + 1].ravel()
            y2 = self.data_for_directions_y[examples]
            x2 = self.data_for_directions_x[examples]

            a = test_categorization(
                y, self.encoder(x), get_category_directions(y2, self.encoder(x2))
            )
            acc.append(a.numpy(force=True))
        return np.mean(acc)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        accuracy = self.validation_accuracy(x, y)
        print("accuracy", accuracy)

        self.log("val_accuracy", accuracy)
