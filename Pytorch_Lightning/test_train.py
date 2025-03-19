import torch
from trainers.train import train_model
from trainers.optimize import optimize_hyperparameters

# Directly calling train_model with parameters
train_model(batch_size=32, epochs=1, log_steps=10)

# Call the optimization function
# optimize_hyperparameters(n_trials=10)