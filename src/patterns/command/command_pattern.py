from patterns.strategy.ClassifierContext import ClassifierContext
from patterns.factory.ClassifierFactory import ClassifierFactory
from abc import ABC, abstractmethod
import preprocessing
import pandas as pd
import os
from datetime import datetime
from patterns.observers.observer import Observer, PreprocessingEvent



# Command interface
class Command(ABC):
    @abstractmethod
    def execute(self):
        pass


# Command for preprocessing data
class PreprocessCommand(Command):
    def __init__(self):
        # Initialize observers for preprocessing
        self._observers = []

    def add_observer(self, observer):
        if observer not in self._observers:
            self._observers.append(observer)

    def notify_observers(self, message):
        # Notify only relevant observers
        for observer in self._observers:
            if hasattr(observer, 'update_preprocessing'):  # Check for preprocessing-specific method
                observer.update_preprocessing(message)

    def execute(self):
        print("Checking for preprocessed files...")
        self.notify_observers("Preprocessing started.")
        run_preprocessing()
        self.notify_observers("Preprocessing completed.")

# Command for running the classifier
class RunClassifierCommand(Command):
    def __init__(self, csv_path: str, choice: str):
        self.csv_path = csv_path
        self.choice = choice
        self._results = None
        self._observers = []  # List of observers

    def add_observer(self, observer):
        if observer not in self._observers:
            self._observers.append(observer)

    def notify_observers(self, email_content, predicted_class, confidence_scores):
        event = {
            "email_content": email_content,
            "predicted_class": predicted_class,
            "confidence_scores": confidence_scores,
        }
        for observer in self._observers:
            observer.update(event)

    def execute(self):
        # Classifier factory provides the strategy based on the user's choice
        classifier_factory = ClassifierFactory()
        strategy = classifier_factory.get_strategy(self.choice)

        # Classify the emails
        print(f"Loading data from {self.csv_path}...")
        self._results = self.classify_emails_from_csv(self.csv_path, strategy)

    def classify_emails_from_csv(self, csv_path: str, strategy):
        # Load the preprocessed CSV
        email_data = pd.read_csv(csv_path)

        # Extract features
        X = email_data.iloc[:, :-1].values

        # Context to use the strategy for prediction
        context = ClassifierContext(strategy)
        predictions = []

        for email_content, x in zip(email_data["email_content"], X):
            predicted_class = context.run_classifier_model([x])[0]
            confidence_scores = dict(
                zip(strategy.classes_, strategy.predict_proba([x])[0])
            )
            self.notify_observers(email_content, predicted_class, confidence_scores)
            predictions.append(predicted_class)

        return predictions

    def get_results(self):
        if self._results is None:
            raise RuntimeError("Command has not been executed yet.")
        return self._results


# Invoker class to execute commands
class Invoker:
    def __init__(self):
        self._commands = []

    def add_command(self, command):
        self._commands.append(command)

    def execute_commands(self):
        for command in self._commands:
            command.execute()
        self._commands.clear()


# Paths for the preprocessed CSV files
purchasing_file = "data/Purchasing_preprocessed.csv"
appgallery_file = "data/AppGallery_preprocessed.csv"
email_file = "data/Emails_preprocessed.csv"


# Run preprocessing script includes check for existing preprocessed files
def run_preprocessing():
    if not os.path.exists(purchasing_file):
        print("Preprocessing Purchasing data...")
        preprocessing.preprocess_data("data/Purchasing.csv")
    if not os.path.exists(appgallery_file):
        print("Preprocessing AppGallery data...")
        preprocessing.preprocess_data("data/AppGallery.csv")
    if not os.path.exists(email_file):
        print("Preprocessing Email data...")
        preprocessing.preprocess_data("data/Emails.csv")
