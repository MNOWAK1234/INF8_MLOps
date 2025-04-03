import bentoml
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define a transform to match the input preprocessing that the model expects
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 images are 32x32
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (example values)
])

def send_image(image_path: str):
    # Load the image
    image = Image.open(image_path)
    
    # Apply the transformations
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension (N, C, H, W)
    
    # Convert tensor to a list format that can be sent via the API
    image_list = image_tensor.numpy().tolist()

    # Send the request to the BentoML service
    with bentoml.SyncHTTPClient("http://localhost:3000") as client:
        try:
            # Send prediction request
            result = client.predict(input_data=image_list)
            print("Prediction Result:", result)
            
            # Display the image using matplotlib
            plt.imshow(image)
            plt.title(f"Predicted Label: {result}")
            plt.show()

        except Exception as e:
            print(f"Error: {e}")

# Path to your image file
image_path = "data/cifar10/test/airplane/14.png"
send_image(image_path)
