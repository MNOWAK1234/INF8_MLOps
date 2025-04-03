import bentoml
import numpy as np

# Create a dummy input image (1 image of shape 3x32x32)
# The values should be in the range [0, 1] (or normalized according to your model's expected input)
input_data = np.random.rand(1, 3, 32, 32).astype(np.float32)  # Shape (1, 3, 32, 32)

# Send prediction request to the running service
with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    try:
        # Make a prediction with the input data
        result = client.predict(input_data=input_data.tolist())  # Convert numpy array to list for API compatibility
        print("Prediction Result:", result)  # Print the predicted class label
    except Exception as e:
        print(f"Error: {e}")
