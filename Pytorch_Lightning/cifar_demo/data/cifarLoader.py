import os
import torch
from torchvision import datasets
from sklearn.model_selection import train_test_split
from PIL import Image

# Function to save images
def save_images(dataset, data, folder_path):
    for idx, (image_array, label) in enumerate(data):
        # Convert the image to a PIL Image
        image = Image.fromarray(image_array)

        class_name = dataset.classes[label]  # Get class label
        class_dir = os.path.join(folder_path, class_name)  # Class-specific folder
        os.makedirs(class_dir, exist_ok=True)  # Create class folder

        image_path = os.path.join(class_dir, f'{idx}.png')  # Image file path
        image.save(image_path)  # Save image

        if idx < 5:  # Print only first few images for confirmation
            print(f"Saved: {image_path}")


def load_cifar():
    # Define directories
    DATA_DIR = './cifar_demo/data/cifar10'

    # Create directories for train, val, and test directly in cifar10
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')
    test_dir = os.path.join(DATA_DIR, 'test')

    # Make sure these directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Download CIFAR-10 dataset (train set)
    dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True)

    # Split dataset into train, validation, and test using sklearn
    # We'll do an 80/10/10 split
    train_data, val_test_data = train_test_split(list(zip(dataset.data, dataset.targets)), test_size=0.2, random_state=42, stratify=dataset.targets)
    val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=42, stratify=[item[1] for item in val_test_data])

    # Save train, val, and test sets
    save_images(dataset, train_data, train_dir)
    save_images(dataset, val_data, val_dir)
    save_images(dataset, test_data, test_dir)

    print(f"All images have been saved in {DATA_DIR}")