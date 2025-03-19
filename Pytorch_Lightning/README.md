## HW 1: Pytorch, Pytorch Lightning, Model monitoring, Hyper-parameter optimization

### Description

This is a demo project to showcase the capabilities of Pytorch Lightning and MLOps tools.

We trained a simple model with two convolutional layers for image classification on the CIFAR-10 dataset.

### Tools used

- Wandb for model monitoring
- Optuna for parameter optimization
- Typer for creating a CLI interface for the project

### Running the project

Clone the project:

```bash
git clone git@github.com:MNOWAK1234/INF8_MLOps.git
```

Move to the `Pytorch_Lightning` directory and install the project with dependencies:

```bash
pip install .
```

Display available options:

```bash
python3 -m cifar_demo --help
```

Download the CIFAR-10 dataset:

```bash
python3 -m cifar_demo load-dataset
```

Train the model on the dataset:

```bash
python3 -m cifar_demo train
```

