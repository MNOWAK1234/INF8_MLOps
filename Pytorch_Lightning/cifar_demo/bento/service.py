import bentoml
import numpy as np

# Define the BentoML runtime environment
demo_image = bentoml.images.PythonImage(python_version="3.11") \
    .python_packages("torch", "torchvision", "mlflow")

@bentoml.service(
    image=demo_image,
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class CIFAR10Classifier:
    # Load the latest registered model from BentoML
    bento_model = bentoml.mlflow.load_model("cifar10_model:latest")

    def __init__(self):
        # self.bento_model.eval()
        # AttributeError: 'PyFuncModel' object has no attribute 'eval'
        # set to eval before (while importing to BentoML)
        pass

    @bentoml.api
    def predict(self, input_data: np.ndarray) -> str:
        """
        Perform inference on an input image array.
        Expects input shape (N, 3, 32, 32) where N is batch size.
        Returns the most probable CIFAR-10 class name.
        """
        # CIFAR-10 class labels
        class_labels = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
        input_data = input_data.astype(np.float32)
        predictions = self.bento_model.predict(input_data)
        most_probable_class_idx = np.argmax(predictions, axis=1)[0]
        return class_labels[most_probable_class_idx]