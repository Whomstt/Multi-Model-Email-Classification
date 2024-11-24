from abc import ABC, abstractmethod
import time
from datetime import datetime
from patterns.strategy.ClassifierStrategy import ClassifierStrategy
from sklearn.base import ClassifierMixin

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
        print("\n=== Timing Decorator ===")
        start_time = time.time()
        model = self._classifier.get_model(email_features)
        end_time = time.time()
        print(f"[Timer] Model retrieval took {end_time - start_time:.2f} seconds")
        print("======================\n")
        return model
        
    def predict(self, email_features):
        print("\n=== Timing Decorator ===")
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