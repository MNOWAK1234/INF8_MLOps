import json
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

# Load image
image_path = "data/cifar10/test/airplane/14.png"
image = Image.open(image_path)

# Transform to match model input
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image_tensor = transform(image).unsqueeze(0)  # (1, 3, 32, 32)
image_list = image_tensor.numpy().tolist()  # Convert to JSON format

# Save to a JSON file
with open("image_payload.json", "w") as f:
    json.dump({"input_data": image_list}, f)

print("Saved JSON file: image_payload.json")
