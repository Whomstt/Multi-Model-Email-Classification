from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any

@dataclass
class ClassificationEvent:
    email_content: str
    predicted_class: str
    confidence_scores: Dict[str, float]
    timestamp: datetime = datetime.now()

class Observer(ABC):
    @abstractmethod
    def update(self, event: ClassificationEvent):
        pass


@dataclass
class PreprocessingEvent:
    """Event to represent preprocessing updates."""
    timestamp: datetime
    message: str

