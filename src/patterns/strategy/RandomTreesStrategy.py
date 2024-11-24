from sklearn.ensemble import RandomTreesEmbedding
import joblib
import os
from patterns.strategy.ClassifierStrategy import ClassifierStrategy
import subprocess


class RandomTreesStrategy(ClassifierStrategy):
    def __init__(self):
        # Define the path to the pre-trained model
        model_path = os.path.join(
            "src",
            "models",
            "random_trees_embedding",
            "random_trees_embedding_model.pkl",
        )
        # Load the pre-trained model from the given path
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """Load the pre-trained model from the specified path."""
        if os.path.exists(model_path):
            model = joblib.load(model_path) 
            print(f"Loaded sgd model from {model_path}")
            return model
        else:
            
            print(f"Model file not found at {model_path}")
            random_trees_embedding_path = os.path.join(
                "src", "models", "random_trees_embedding", "random_trees_embedding.py"
            )
            print(f"Running {random_trees_embedding_path} to create the model...")
            subprocess.run(["python", random_trees_embedding_path], check=True)
            model = joblib.load(model_path)
            return model

    def get_model(self, email_features) -> RandomTreesEmbedding:
        """Return the loaded Random Trees Embedding model."""
        return self.model

    def predict(self, X):
        """Make predictions using the loaded model."""
        return self.model.predict(X)
