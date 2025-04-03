import bentoml
import numpy as np

# Define the BentoML runtime environment
demo_image = bentoml.images.PythonImage(python_version="3.11") \
    .python_packages("torch", "torchvision", "mlflow")

# CIFAR-10 class labels
class_labels = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

@bentoml.service(
    image=demo_image,
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class CIFAR10Classifier:
    # Load the latest registered model from BentoML
    bento_model = bentoml.mlflow.load_model("cifar10_model:latest")

    def __init__(self):
        # No need to call eval() for PyFuncModel, it's managed by BentoML
        pass

    @bentoml.api
    def predict(self, input_data: np.ndarray) -> str:
        """
        Perform inference on an input image array.
        Expects input shape (N, 3, 32, 32) where N is batch size.
        Returns the most probable CIFAR-10 class name.
        """
        # Prepare input data for prediction (ensure correct dtype)
        input_data = input_data.astype(np.float32)

        # Perform inference with the model's `predict` method
        predictions = self.bento_model.predict(input_data)

        # Find the index of the class with the highest score
        most_probable_class_idx = np.argmax(predictions, axis=1)[0]

        # Get the class label corresponding to the index
        return class_labels[most_probable_class_idx]
