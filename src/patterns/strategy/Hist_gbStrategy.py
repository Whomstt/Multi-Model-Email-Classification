from sklearn.ensemble import (
    HistGradientBoostingClassifier,
)  
import joblib
import os
from patterns.strategy.ClassifierStrategy import ClassifierStrategy
import subprocess


class Hist_gbStrategy(ClassifierStrategy):
    def __init__(self):
        # Define the path to the pre-trained model
        model_path = os.path.join("src", "models", "hist_gb", "hist_gb_model.pkl")
        # Load the pre-trained model from the given path
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """Load the pre-trained model from the specified path."""
        if os.path.exists(model_path):
            model = joblib.load(model_path)  
            print(f"Loaded hist_gb model from {model_path}")
            return model
        else:
            # If the model file does not exist, create it
            print(f"Model file not found at {model_path}")
            hist_gb_path = os.path.join("src", "models", "hist_gb", "hist_gb.py")
            print(f"Running {hist_gb_path} to create the model...")
            subprocess.run(["python", hist_gb_path], check=True)
            model = joblib.load(model_path)
            return model

    def get_model(self, email_features) -> HistGradientBoostingClassifier:
        """Return the loaded hist_gb model."""
        return self.model

    def predict(self, X):
        """Make predictions using the loaded model."""
        return self.model.predict(X)
