from patterns.strategy.ClassifierStrategy import ClassifierStrategy

class ClassifierContext:
    def __init__(self, strategy: ClassifierStrategy):
        if not isinstance(strategy, ClassifierStrategy):
            raise ValueError("strategy must be an instance of ClassifierStrategy")
        self._strategy = strategy  # Store strategy as _strategy

    def run_classifier_model(self, email_features):
        """Run the classifier model using the selected strategy and pass the email features."""
        print(f"Running model using {self._strategy.__class__.__name__}")
        
        # Access _strategy, not strategy
        model = self._strategy.get_model(email_features)  # Assuming get_model doesn't need features
        prediction = self._strategy.predict(email_features)  # Pass email_features to predict
        return prediction
