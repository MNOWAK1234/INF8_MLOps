import torch
import torchvision.transforms as transforms
from PIL import Image
import os

import sys
from pathlib import Path

# Add the path to the directory containing your model script
sys.path.append(str(Path("../Pytorch_Lightning/cifar_demo/models").resolve()))

from model import CIFAR10Model  # Import your model class

# Define CIFAR-10 class names
cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Load the model
model = CIFAR10Model()
model.load_state_dict(torch.load("cifar10_model.pth"))
model.eval()  # Set to evaluation mode

# Preprocessing function
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 images are 32x32
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Path to CIFAR-10 test dataset
dataset_path = "../Pytorch_Lightning/cifar_demo/data/cifar10/test/"

# Variables to track accuracy
total_images = 0
correct_predictions = 0
class_counts = {cls: 0 for cls in cifar10_classes}  # Total images per class
correct_classifications = {cls: 0 for cls in cifar10_classes}  # Correct predictions per class

# Iterate over each class folder
for class_name in cifar10_classes:
    class_path = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_path):
        continue  # Skip if not a directory

    for filename in os.listdir(class_path):
        if not filename.endswith((".png", ".jpg", ".jpeg")):
            continue  # Skip non-image files

        # Load image
        image_path = os.path.join(class_path, filename)
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # Preprocess

        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()

        # Update counters
        total_images += 1
        class_counts[class_name] += 1
        if cifar10_classes[predicted_class] == class_name:
            correct_predictions += 1
            correct_classifications[class_name] += 1

# Print class-wise results
print("\nClassification Results:")
for cls in cifar10_classes:
    total = class_counts[cls]
    correct = correct_classifications[cls]
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"  {cls}: {correct}/{total} correctly classified ({accuracy:.2f}%)")

# Print overall accuracy
overall_accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
print(f"\nOverall Accuracy: {overall_accuracy:.2f}% ({correct_predictions}/{total_images})")
