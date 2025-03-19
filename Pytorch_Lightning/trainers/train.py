import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from data.datamodules.cifar_datamodule import CIFAR10DataModule
from models.model import CIFAR10Model

def train_model(batch_size=32, epochs=10, log_steps=10):
    # Instantiate the DataModule and Model with passed parameters
    datamodule = CIFAR10DataModule(batch_size=batch_size)
    model = CIFAR10Model()

    # Checkpoint callback to save best model
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="best_model",
        save_top_k=1,
        mode="min"
    )

    # Instantiate the Trainer
    trainer = Trainer(
        max_epochs=epochs,
        devices=1,  # Set to 1 for CPU or GPU
        accelerator="gpu" if torch.cuda.is_available() else "cpu",  # Automatically use GPU or CPU
        log_every_n_steps=log_steps,
        callbacks=[checkpoint_callback]
    )

    # Train the model
    trainer.fit(model, datamodule=datamodule)

    # Validate before testing
    trainer.validate(model, datamodule=datamodule)

    # Test the model
    trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    # Call the function with parameters
    train_model(batch_size=32, epochs=10, log_steps=1)
