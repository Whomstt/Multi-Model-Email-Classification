from sklearn.linear_model import SGDClassifier
import joblib
import os
from patterns.strategy.ClassifierStrategy import ClassifierStrategy
import subprocess


class SgdStrategy(ClassifierStrategy):
    def __init__(self):
        # Define the path to the pre-trained model
        model_path = os.path.join("src", "models", "sgd", "sgd_model.pkl")
        # Load the pre-trained model from the given path
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """Load the pre-trained model from the specified path."""
        if os.path.exists(model_path):
            model = joblib.load(model_path)  
            print(f"Loaded sgd model from {model_path}")
            return model
        else:
            # If the model file does not exist, create it
            print(f"Model file not found at {model_path}")
            sgd_path = os.path.join("src", "models", "sgd", "sgd.py")
            print(f"Running {sgd_path} to create the model...")
            subprocess.run(["python", sgd_path], check=True)
            model = joblib.load(model_path)
            return model

    def get_model(self, email_features) -> SGDClassifier:
        """Return the loaded SGD model."""
        return self.model

    def predict(self, X):
        """Make predictions using the loaded model."""
        return self.model.predict(X)
