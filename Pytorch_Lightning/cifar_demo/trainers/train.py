# trainers/train.py
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from cifar_demo.data.datamodules.cifar_datamodule import CIFAR10DataModule
from cifar_demo.models.model import CIFAR10Model

def train_model(batch_size=32, epochs=10, log_steps=10, learning_rate=1e-3):
    """
    Function to train the model with given hyperparameters.
    """

    # Prepare the DataModule and Model
    datamodule = CIFAR10DataModule(batch_size=batch_size)
    model = CIFAR10Model()

    # Checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", 
        filename="best_model", 
        save_top_k=1, 
        mode="min"
    )

    # Wandb Logger setup
    wandb_logger = WandbLogger(project="MLOps_lab1", log_model=True)

    # Initialize the Trainer
    trainer = Trainer(
        max_epochs=epochs,
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=log_steps,
        callbacks=[checkpoint_callback],
        logger=wandb_logger
    )

    # Configure optimizer with learning rate
    model.configure_optimizers = lambda: torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    trainer.fit(model, datamodule=datamodule)

    # Validate the model after training
    trainer.validate(model, datamodule=datamodule)

    # Accessing the validation loss after validation phase
    loss = trainer.callback_metrics["val_loss"].item()

    # Test the model after validation
    trainer.test(model, datamodule=datamodule)

    # # Return test loss for hyperparameter optimization
    # loss = trainer.callback_metrics["test_loss"].item()

    return loss
