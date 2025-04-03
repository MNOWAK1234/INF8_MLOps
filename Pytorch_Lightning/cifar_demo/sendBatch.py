import bentoml
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt

# Define a transform to match the input preprocessing that the model expects
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 images are 32x32
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (example values)
])

def send_batch(image_paths: list):
    # Load and preprocess all images in the batch
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension (N, C, H, W)
        images.append(image_tensor)
    
    # Stack all images into one batch tensor
    batch_tensor = torch.cat(images, dim=0)
    
    # Convert to list format for sending
    batch_list = batch_tensor.numpy().tolist()

    # Send the batch to the BentoML service
    with bentoml.SyncHTTPClient("http://localhost:3000") as client:
        try:
            result = client.predict(input_data=batch_list)
            print("Prediction Results:", result)
            
            # Display the images and their predictions
            for i, image_path in enumerate(image_paths):
                image = Image.open(image_path)
                plt.imshow(image)
                plt.title(f"Predicted Label: {result[i]}")
                plt.show()

        except Exception as e:
            print(f"Error: {e}")

# Batch of image paths
image_paths = [
    "data/cifar10/test/airplane/14.png",
    "data/cifar10/test/airplane/29.png",  # Add more image paths as needed
    "data/cifar10/test/automobile/3.png"
]
send_batch(image_paths)
