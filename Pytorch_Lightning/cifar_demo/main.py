import typer
import subprocess

from cifar_demo.data.cifarLoader import load_cifar
from cifar_demo.trainers.train import train_model
from cifar_demo.trainers.optimize import optimize_hyperparameters
from cifar_demo.models.export.toPytorch import export_model_to_pytorch
from cifar_demo.bento.bento_import import import_model_to_bentoml

app = typer.Typer()

@app.command()
def load_dataset():
    """
    Load the CIFAR-10 dataset.
    """
    load_cifar()

@app.command()
def train(save_dir: str = typer.Option(None, help="Optional directory to save the model")):
    """
    Train a model on the dataset. If no save_dir is provided, it defaults to WandB's run ID.
    """
    train_model(batch_size=32, epochs=10, log_steps=1, save_dir=save_dir)

@app.command()
def optimize():
    """
    Optimize model hyperparameters with Optuna.
    """
    optimize_hyperparameters(n_trials=10)

@app.command()
def export_to_pytorch(checkpoint_path: str = typer.Option("my_experiment/best_model.ckpt", help="Path to model checkpoint (probably [name]/best_model.ckpt)"),
                      output_path: str = typer.Option("cifar10_model.pth", help="Path to save the .pth file")):
    """
    Export the trained model to PyTorch format (.pth).
    """
    export_model_to_pytorch(checkpoint_path, output_path)


@app.command()
def import_to_bentoml(checkpoint_path: str = typer.Option("checkpoints/best_model.ckpt", help="Path to model checkpoint (probably [name]/best_model.ckpt)"),
                       model_name: str = typer.Option("cifar10_model", help="Name for the BentoML model")):
    """
    Import the trained PyTorch model into BentoML.
    """
    import_model_to_bentoml(checkpoint_path, model_name)

@app.command()
def serve():
    """
    Host the BentoML model as a service.
    """
    subprocess.run(["bentoml", "serve", "cifar_demo.bento.service:CIFAR10Classifier"], check=True)

if __name__ == "__main__":
    app()
