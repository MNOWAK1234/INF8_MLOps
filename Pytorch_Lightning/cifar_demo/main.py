import typer

from cifar_demo.data.cifarLoader import load_cifar
from cifar_demo.trainers.train import train_model
from cifar_demo.trainers.optimize import optimize_hyperparameters

app = typer.Typer()

@app.command()
def load_dataset():
    """
    Load the CIFAR-10 dataset.
    """
    load_cifar()

@app.command()
def train():
    """
    Train a model on the dataset.
    """
    train_model(batch_size=32, epochs=10, log_steps=1)

@app.command()
def optimize():
    """
    Optimize model hyperparameters with Optuna.
    """
    optimize_hyperparameters(n_trials=10)

if __name__ == "__main__":
    app()

