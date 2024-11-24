from patterns.strategy.ClassifierContext import ClassifierContext
from patterns.factory.ClassifierFactory import ClassifierFactory
from patterns.strategy.ClassifierStrategy import ClassifierStrategy
from abc import ABC, abstractmethod
import preprocessing
import numpy as np
import os
import pandas as pd
from patterns.decorator.classifier_decorator import (
     LoggingDecorator,
    TimingDecorator,
    ValidationDecorator,
    StatisticsDecorator,
    ErrorHandlingDecorator,
    DataNormalizationDecorator,
    
    
)



# Command interface
class Command(ABC):
    @abstractmethod
    def execute(self):
        pass


# Command for preprocessing data
class PreprocessCommand(Command):
    def execute(self):
        print("Checking for preprocessed files...")
        run_preprocessing()


# Command for running the classifier
class RunClassifierCommand(Command):
    def __init__(self, csv_path: str, choice: str):
        self.csv_path = csv_path
        self.choice = choice
        self._results = None

    def execute(self):
        # Classifier factory provides the strategy based on the user's choice
        classifier_factory = ClassifierFactory()
        strategy = classifier_factory.get_strategy(self.choice)

        # Classify the emails
        print(f"Loading data from {self.csv_path}...")
        self._results = self.classify_emails_from_csv(self.csv_path, strategy)

        # Only modify the classify_emails_from_csv method in RunClassifierCommand


    def classify_emails_from_csv(self, csv_path: str, strategy):
        # Load the preprocessed CSV
        email_data = pd.read_csv(csv_path)

        # Extract features
        X = email_data.iloc[:, :-1].values

        decorated_strategy = ValidationDecorator(strategy)
        decorated_strategy = DataNormalizationDecorator(decorated_strategy)
        decorated_strategy = ErrorHandlingDecorator(decorated_strategy)
        decorated_strategy = StatisticsDecorator(decorated_strategy)
        decorated_strategy = TimingDecorator(decorated_strategy)
        decorated_strategy = LoggingDecorator(decorated_strategy)
        # Create context with decorated strategy
        context = ClassifierContext(decorated_strategy)
        predictions = context.run_classifier_model(X)


        return predictions

    # Get the results of the command
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
