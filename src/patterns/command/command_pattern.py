from patterns.strategy.ClassifierContext import ClassifierContext
from patterns.factory.ClassifierFactory import ClassifierFactory
from patterns.strategy.ClassifierStrategy import ClassifierStrategy
from abc import ABC, abstractmethod
import preprocessing
import numpy as np
import os
import pandas as pd


class Command(ABC):
    @abstractmethod
    def execute(self):
        pass

class PreprocessCommand(Command):
    def execute(self):
        print("Checking for preprocessed files...")
        run_preprocessing()

class RunClassifierCommand(Command):
    def __init__(self, choice: str):
        self.choice = choice

    def execute(self):
        classifier_factory = ClassifierFactory()
        strategy = classifier_factory.get_strategy(self.choice)
        context = ClassifierContext(strategy)
        context.run_classifier_model()

    def classify_emails_from_csv(self, csv_path: str, strategy):
       """Classify emails from a preprocessed CSV file."""
       import pandas as pd  # Import pandas to read the CSV

       # Load the preprocessed CSV
       print(f"Loading data from {csv_path}...")
       email_data = pd.read_csv(csv_path)

       # Extract features (assume all columns except the last are features)
       X = email_data.iloc[:, :-1].values

       # Use the strategy to predict
       context = ClassifierContext(strategy)
       predictions = context.run_classifier_model(X)

       return predictions


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

# Run preprocessing script
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
