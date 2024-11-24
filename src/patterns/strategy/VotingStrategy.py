from sklearn.ensemble import VotingClassifier
import joblib
import os
from patterns.strategy.ClassifierStrategy import ClassifierStrategy
import subprocess


class VotingStrategy(ClassifierStrategy):
    def __init__(self):
        # Define the path to the pre-trained model
        model_path = os.path.join("src", "models", "voting", "voting_model.pkl")
        # Load the pre-trained model from the given path
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """Load the pre-trained model from the specified path."""
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"Loaded voting model from {model_path}")
            return model
        else:
            # If the model file does not exist, create it
            print(f"Model file not found at {model_path}")
            voting_script_path = os.path.join("src", "models", "voting", "voting.py")
            print(f"Running {voting_script_path} to create the model...")
            subprocess.run(["python", voting_script_path], check=True)
            model = joblib.load(model_path)
            return model

    def get_model(self, email_features) -> VotingClassifier:
        """Return the loaded voting model."""
        return self.model

    def predict(self, X):
        """Make predictions using the loaded model."""
        return self.model.predict(X)
