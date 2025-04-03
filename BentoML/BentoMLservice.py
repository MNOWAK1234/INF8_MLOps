import torch
import bentoml
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from bentoml.models import BentoModel
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Define the runtime environment for your Bento
demo_image = bentoml.images.PythonImage(python_version="3.11") \
    .python_packages("torch", "torchvision", "Pillow")

# CIFAR-10 class labels
cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

@bentoml.service(
    image=demo_image,
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)

class CIFAR10Classifier:
    bento_model = BentoModel("cifar10_model:latest")

    def __init__(self):
        logging.info("Loading the model...")
        self.model = bentoml.mlflow.load_model(self.bento_model)
        logging.info("Model loaded successfully!")

    @bentoml.api
    def predict(self, input_data: np.ndarray) -> list[str]:
        logging.info("Received prediction request...")
        image = Image.fromarray(input_data)
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        
        logging.info(f"Prediction: {cifar10_classes[predicted_class]}")
        return [cifar10_classes[predicted_class]]

