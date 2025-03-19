import torch
from trainers.train import train_model

# Directly calling train_model with parameters
train_model(batch_size=32, epochs=10, log_steps=10)
