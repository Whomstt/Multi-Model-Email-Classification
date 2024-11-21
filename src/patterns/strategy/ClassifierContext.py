
from patterns.strategy.ClassifierStrategy import ClassifierStrategy

class ClassifierContext:
    def __init__(self, strategy: ClassifierStrategy):
        if not isinstance(strategy, ClassifierStrategy):
            raise ValueError("strategy must be an instance of ClassifierStrategy")
        self._strategy = strategy

    def run_classifier_model(self):
        """Run the classifier model using the selected strategy."""
        print(f"Running model using {self._strategy.__class__.__name__}")
        self._strategy.get_model()
