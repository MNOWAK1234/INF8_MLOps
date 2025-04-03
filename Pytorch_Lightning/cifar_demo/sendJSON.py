import requests
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import json

image_path = "data/cifar10/test/airplane/14.png"
image = Image.open(image_path)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_tensor = transform(image).unsqueeze(0)  # Shape (1, 3, 32, 32)

image_list = image_tensor.numpy().tolist()  # Shape: (1, 3, 32, 32)
data = json.dumps({"input_data": image_list})  # No extra brackets!

url = "http://localhost:3000/predict"
headers = {"Content-Type": "application/json"}
response = requests.post(url, headers=headers, data=data)

if response.status_code == 200:
    print("Prediction:", response.text)
else:
    print("Error:", response.status_code, response.text)
