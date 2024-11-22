from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
import os
from patterns.strategy.ClassifierStrategy import ClassifierStrategy
import subprocess


class AdaBoostStrategy(ClassifierStrategy):  # Ensure it inherits from ModelStrategy
    def __init__(self):
        model_path = os.path.join("src", "models", "adaboosting", "adaboost_model.pkl")
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """Load the pre-trained model from the specified path."""
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"Loaded AdaBoost model from {model_path}")
            return model
        else:
            # If the model file does not exist, create it
            print(f"Model file not found at {model_path}")
            adaboosting_path = os.path.join(
                "src", "models", "adaboosting", "adaboosting.py"
            )
            print(f"Running {adaboosting_path} to create the model...")
            subprocess.run(["python", adaboosting_path], check=True)
            model = joblib.load(model_path)
            return model

    def get_model(self) -> AdaBoostClassifier:
        """Return the loaded AdaBoost model."""
        return self.model

    def predict(self, X):
        """Make predictions using the loaded model."""
        return self.model.predict(X)
