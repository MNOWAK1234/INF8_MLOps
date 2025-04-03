import requests
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import json

# Prepare the image (preprocess as expected by the model)
image_path = "data/cifar10/test/airplane/14.png"
image = Image.open(image_path)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Convert tensor to list format
image_list = image_tensor.squeeze(0).numpy().tolist()  # Shape [3, 32, 32]

# Send the request to the BentoML service
url = "http://localhost:3000/predict"
headers = {"Content-Type": "application/json"}
data = json.dumps({"input_data": [image_list]})

response = requests.post(url, headers=headers, data=data)

# Display the prediction
if response.status_code == 200:
    print("Prediction:", response)
else:
    print("Error:", response.status_code, response.text)
