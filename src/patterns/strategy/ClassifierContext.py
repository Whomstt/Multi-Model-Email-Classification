
from patterns.strategy.ClassifierStrategy import ClassifierStrategy

class ClassifierContext:
    def __init__(self, strategy):
        self._strategy = strategy

    def run_classifier_model(self):
        """Run the classifier model using the selected strategy."""
        print("Running model using", self._strategy.__class__.__name__)
        self._strategy.get_model()  # You could add more functionality like .predict(), etc.
