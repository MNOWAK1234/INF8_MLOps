import torch
from cifar_demo.models.model import CIFAR10Model

# Create the model
model = CIFAR10Model()

# Fake input data: batch of 4 CIFAR-10 images (3 channels, 32x32)
dummy_input = torch.randn(4, 3, 32, 32)

# Forward pass test
output = model(dummy_input)

# Check output shape (should be [4, 10] for 4 images and 10 classes)
print("Output shape:", output.shape)
