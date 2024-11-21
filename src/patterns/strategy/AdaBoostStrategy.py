from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

class AdaBoostStrategy(ModelStrategy):  # Ensure it inherits from ModelStrategy
    def __init__(self):
        """Initialize by loading the pre-trained AdaBoost model from file."""
        model_path = os.path.join("models", "adaboosting", "adaboost_model.pkl")
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """Load the pre-trained model from the specified path."""
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"Loaded AdaBoost model from {model_path}")
            return model
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")

    def get_model(self) -> AdaBoostClassifier:
        """Return the loaded AdaBoost model."""
        return self.model

    def predict(self, X):
        """Make predictions using the loaded model."""
        return self.model.predict(X)