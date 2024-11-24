from abc import ABC, abstractmethod
import time
from datetime import datetime
from patterns.strategy.ClassifierStrategy import ClassifierStrategy
from sklearn.base import ClassifierMixin
import numpy as np


class ClassifierDecorator(ClassifierStrategy):
    """Base decorator that implements ClassifierStrategy"""

    def __init__(self, classifier: ClassifierStrategy):
        self._classifier = classifier

    def get_model(self, email_features) -> ClassifierMixin:
        """Pass through the get_model call to decorated classifier"""
        return self._classifier.get_model(email_features)

    def predict(self, email_features):
        """Pass through the predict call to decorated classifier"""
        return self._classifier.predict(email_features)
    
    


class LoggingDecorator(ClassifierDecorator):
    def get_model(self, email_features) -> ClassifierMixin:
        print("\n=== Logging Decorator ===")
        print(f"[Log] Getting model at: {datetime.now()}")
        model = self._classifier.get_model(email_features)
        print(f"[Log] Model retrieved at: {datetime.now()}")
        print("========================\n")
        return model

    def predict(self, email_features):
        print("\n=== Logging Decorator ===")
        print(f"[Log] Starting prediction at: {datetime.now()}")
        predictions = self._classifier.predict(email_features)
        print(f"[Log] Prediction completed at: {datetime.now()}")
        print("========================\n")
        return predictions
    


class TimingDecorator(ClassifierDecorator):
    def get_model(self, email_features) -> ClassifierMixin:
        start_time = time.time()
        model = self._classifier.get_model(email_features)
        end_time = time.time()
        print(f"[Timer] Model retrieval took {end_time - start_time:.2f} seconds")
        print("======================\n")
        return model

    def predict(self, email_features):
        start_time = time.time()
        predictions = self._classifier.predict(email_features)
        end_time = time.time()
        print(f"[Timer] Prediction took {end_time - start_time:.2f} seconds")
        print("======================\n")
        return predictions
    


class ValidationDecorator(ClassifierDecorator):
    def get_model(self, email_features) -> ClassifierMixin:
        print("\n=== Validation Decorator ===")
        if email_features is None:
            raise ValueError("[Validation] Error: No email features provided!")
        print(f"[Validation] Processing features with shape: {email_features.shape}")
        model = self._classifier.get_model(email_features)
        if model is None:
            raise ValueError("[Validation] Error: Model is None!")
        print("[Validation] Model validated successfully")
        print("===========================\n")
        return model

    def predict(self, email_features):
        print("\n=== Validation Decorator ===")
        if email_features is None:
            raise ValueError("[Validation] Error: No email features provided!")
        print(f"[Validation] Processing features with shape: {email_features.shape}")
        predictions = self._classifier.predict(email_features)
        print("[Validation] Predictions generated successfully")
        print("===========================\n")
        return predictions
    


class StatisticsDecorator(ClassifierDecorator):
    """Adds statistical analysis of predictions"""

    def predict(self, email_features):
        predictions = self._classifier.predict(email_features)

        # Calculate distribution of predictions
        unique, counts = np.unique(predictions, return_counts=True)
        distribution = dict(zip(unique, counts))

        print("[Stats] Prediction Distribution:")
        for category, count in distribution.items():
            percentage = (count / len(predictions)) * 100
            print(f"- {category}: {count} emails ({percentage:.1f}%)")
        print("=========================\n")
        return predictions
    


class ErrorHandlingDecorator(ClassifierDecorator):
    """Adds robust error handling"""

    def predict(self, email_features):
        try:
            if not isinstance(email_features, np.ndarray):
                raise ValueError("Input must be a numpy array")

            predictions = self._classifier.predict(email_features)
            print("[Error Handler] Prediction completed successfully")
            print("=============================\n")
            return predictions

        except Exception as e:
            print(f"[Error Handler] Error occurred: {str(e)}")
            print("=============================\n")
            return ["Error in prediction"] * len(email_features)
         
   


class DataNormalizationDecorator(ClassifierDecorator):
    """Adds data normalization"""

    def predict(self, email_features):
        print("\n=== Normalization Decorator ===")
        normalized_features = (
            (email_features - np.min(email_features))
            / (np.max(email_features) - np.min(email_features))
        )
        print("[Normalization] Features normalized to [0,1] range")
        print("==============================\n")
        return self._classifier.predict(normalized_features)
     
    
