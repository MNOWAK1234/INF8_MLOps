import torch
from cifar_demo.models.model import CIFAR10Model

def export_model_to_pytorch(checkpoint_path="my_experiment/best_model.ckpt", output_path="cifar10_model.pth"):
    """
    Load a trained model from a checkpoint and save it as a PyTorch .pth file.
    """
    model = CIFAR10Model.load_from_checkpoint(checkpoint_path)
    model.eval()
    torch.save(model.state_dict(), output_path)
    print(f"Model saved as {output_path}")