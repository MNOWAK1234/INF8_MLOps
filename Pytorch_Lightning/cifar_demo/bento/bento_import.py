import mlflow.pytorch
import bentoml
from pathlib import Path
import torch
from cifar_demo.models.model import CIFAR10Model

def import_model_to_bentoml(checkpoint_path="my_experiment/best_model.ckpt", model_name="cifar10_model"):
    """
    Import a trained PyTorch model into BentoML.
    """
    model = CIFAR10Model.load_from_checkpoint(checkpoint_path)
    model.eval() # Do it here to avoid AttributeError: 'PyFuncModel' object has no attribute 'eval' in service
    model_uri = Path("models", model_name)
    mlflow.pytorch.save_model(model, model_uri.resolve())
    bentoml.mlflow.import_model(model_name, model_uri)

    print(f"Model imported into BentoML as '{model_name}'")
