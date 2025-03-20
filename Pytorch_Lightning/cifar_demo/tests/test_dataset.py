from cifar_demo.data.datasets.cifar_dataset import CIFAR10Dataset
from PIL import Image

# Class labels for CIFAR-10
class_names = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 
    'Frog', 'Horse', 'Ship', 'Truck'
]

# Initialize dataset
dataset = CIFAR10Dataset(root_dir="data/cifar10/train", transform=None)

# Test the Dataset by loading a single sample
sample_image, sample_label = dataset[0]  # Get the first sample

# Show image and label
print(f"Label: {sample_label} - Class: {class_names[sample_label]}")
sample_image.show()  # Show the image

# Check the image type (should be PIL Image)
print(f"Image Type: {type(sample_image)}")
