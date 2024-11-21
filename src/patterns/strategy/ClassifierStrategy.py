from abc import ABC, abstractmethod
from sklearn.base import ClassifierMixin

class ClassifierStrategy(ABC):
    @abstractmethod
    def get_model(self) -> ClassifierMixin:
        """Return the pre-trained classifier model."""
        pass