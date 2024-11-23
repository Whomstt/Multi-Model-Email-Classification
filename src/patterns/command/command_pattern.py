from abc import ABC, abstractmethod
from patterns.strategy.ClassifierContext import ClassifierContext
from patterns.factory.ClassifierFactory import ClassifierFactory
import preprocessing
import os
import sys
import subprocess


class Command(ABC):
    @abstractmethod
    def execute(self):
        pass


# Run preprocessing
class PreprocessCommand(Command):
    def execute(self):
        print("Checking for preprocessed files...")
        run_preprocessing()


# Run a model using the strategy
class RunClassifierCommand(Command):
    def __init__(self, choice):
        self.choice = choice

    def execute(self):
        classifier_factory = ClassifierFactory()
        strategy = classifier_factory.get_strategy(self.choice)
        context = ClassifierContext(strategy)
        context.run_classifier_model()


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
email_file = "data/Email_preprocessed.csv"


# Run preprocessing script
def run_preprocessing():
    if not os.path.exists(purchasing_file):
        print("Preprocessing Purchasing data...")
        preprocessing.preprocess_data("data/Purchasing.csv")
    elif not os.path.exists(appgallery_file):
        print("Preprocessing AppGallery data...")
        preprocessing.preprocess_data("data/AppGallery.csv")
    elif not os.path.exists(email_file):
        print("Preprocessing Email data...")
        preprocessing.preprocess_data("data/Email.csv")
