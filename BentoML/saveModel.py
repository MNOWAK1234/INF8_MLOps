import torch
from ../Pytorch_Lightning/cifar_demo/models/model.py import CIFAR10Model  # Import your model class

# Load model from checkpoint
model = CIFAR10Model.load_from_checkpoint("../Pytorch_Lightning/MLOps_lab1/hqlc19ds/checkpoints/best_model.ckpt")
model.eval()  # Set to evaluation mode

# Save the model's state dictionary
torch.save(model.state_dict(), "cifar10_model.pth")

print("Model saved as cifar10_model.pth")
