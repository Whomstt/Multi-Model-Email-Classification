from patterns.observers.observer import Observer, ClassificationEvent
from dataclasses import dataclass
from datetime import datetime
import os
from patterns.observers.observer import Observer, PreprocessingEvent, ClassificationEvent



class ExecutionLogger(Observer):
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "preprocessing_log.txt")

    def update_preprocessing(self, message):
        # Log the preprocessing message to a file
        with open(self.log_file, "a") as f:
            f.write(f"{message}\n")

    def update(self, event):
        # Handle classification events
        with open(self.log_file, "a") as f:
            f.write(f"{event.timestamp}: {event.predicted_class} - {event.confidence_scores}\n")

