from sklearn.ensemble import HistGradientBoostingClassifier  # Import the correct classifier
import joblib
import os
from patterns.strategy.ClassifierStrategy import ClassifierStrategy

class Hist_gbStrategy(ClassifierStrategy):  
    def __init__(self):
        # Define the path to the pre-trained model
        model_path = os.path.join("src","models", "hist_gb", "hist_gb_model.pkl")
        # Load the pre-trained model from the given path
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """Load the pre-trained model from the specified path."""
        if os.path.exists(model_path):
            model = joblib.load(model_path)  # Load the model using joblib
            print(f"Loaded hist_gb model from {model_path}")
            return model
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")

    def get_model(self) -> HistGradientBoostingClassifier:
        """Return the loaded hist_gb model."""
        return self.model

    def predict(self, X):
        """Make predictions using the loaded model."""
        return self.model.predict(X)
