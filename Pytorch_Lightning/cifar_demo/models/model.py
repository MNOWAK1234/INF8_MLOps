import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.classification import Accuracy  # Track accuracy

class CIFAR10Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Define the layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for CIFAR-10

        # Accuracy metric for multiclass classification
        self.accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        # Forward pass through the network
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        # Training step: receive batch, calculate loss, and log metrics
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        acc = self.accuracy(outputs, labels)

        # Log loss and accuracy to the progress bar
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss  # Return the loss so PyTorch Lightning can optimize it

    def validation_step(self, batch, batch_idx):
        # Validation step: receive batch, calculate loss and accuracy
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        acc = self.accuracy(outputs, labels)

        # Log loss and accuracy for validation
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss  # Return the loss for optimization purposes

    def test_step(self, batch, batch_idx):
        # Test step: receive batch, calculate loss and accuracy
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        acc = self.accuracy(outputs, labels)

        # Log test loss and accuracy
        self.log("test_loss", loss)
        self.log("test_acc", acc)

        return loss  # Return the loss

    def configure_optimizers(self):
        # Set up the optimizer
        return torch.optim.Adam(self.parameters(), lr=1e-3)
