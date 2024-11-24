from patterns.observers.observer import Observer, ClassificationEvent

class ResultsDisplayer(Observer):
    def update(self, event: ClassificationEvent):
        print(f"Email: {event.email_content[:50]}...")
        print(f"Predicted: {event.predicted_class}")
        print(f"Confidence: {event.confidence_scores}")
