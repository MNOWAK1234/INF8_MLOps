# trainers/optimize.py
import optuna
from trainers.train import train_model  # Import the train_model function

def objective(trial):
    """
    Optuna objective function for hyperparameter optimization.
    """
    # Suggest hyperparameters using Optuna
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    batch_size = trial.suggest_int("batch_size", 16, 128, step=16)

    # Train the model with suggested hyperparameters and get the validation loss
    val_loss = train_model(batch_size=batch_size, epochs=3, log_steps=10, learning_rate=learning_rate)

    return val_loss

def optimize_hyperparameters(n_trials=10):
    """
    Function to optimize hyperparameters using Optuna.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Print the best trial after optimization
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (validation loss): {trial.value}")
    print(f"  Params: {trial.params}")
