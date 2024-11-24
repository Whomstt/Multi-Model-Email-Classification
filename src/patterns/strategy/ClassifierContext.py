from patterns.strategy.ClassifierStrategy import ClassifierStrategy

class ClassifierContext:
    def __init__(self, strategy: ClassifierStrategy):
    # Ensure the provided strategy is an instance of ClassifierStrategy.
        if not isinstance(strategy, ClassifierStrategy):
            raise ValueError("strategy must be an instance of ClassifierStrategy")
        self._strategy = strategy  

    def run_classifier_model(self, email_features):
        """Run the classifier model using the selected strategy and pass the email features."""
        print(f"Running model using {self._strategy.__class__.__name__}")
        
        # Get the model selected and return the prediction
        model = self._strategy.get_model(email_features) 
        prediction = self._strategy.predict(email_features) 
        return prediction
