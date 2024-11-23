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


# Run a specific script
class RunScriptCommand(Command):
    def __init__(self, script_path):
        self.script_path = script_path

    def execute(self):
        run_script(self.script_path)


# Run a model using the strategy
class RunClassifierCommand(Command):
    def __init__(self, choice):
        self.choice = choice

    def execute(self):
        if model_exists(self.choice):
            print("Model already exists. Skipping model training.")
        else:
            print("Model does not exist. Training model...")
            run_script(model_script(self.choice))
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


# Run preprocessing script
def run_preprocessing():
    if not os.path.exists(purchasing_file):
        print("Preprocessing Purchasing data...")
        preprocessing.preprocess_data("data/Purchasing.csv")
    elif not os.path.exists(appgallery_file):
        print("Preprocessing AppGallery data...")
        preprocessing.preprocess_data("data/AppGallery.csv")


# Run our model scripts
def run_script(script_path):
    print(f"Running script: {script_path}")
    if not os.path.exists(script_path):
        print(f"Error: '{script_path}' not found.")
        return

    try:
        result = subprocess.run([sys.executable, script_path], check=True)
        print(f"Finished {script_path} with code {result.returncode}.")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


# Check if the model exists
def model_exists(choice):
    # Map choices to corresponding model files
    model_files = {
        "1": "src/models/adaboosting/adaboost_model.pkl",
        "2": "src/models/voting/voting_model.pkl",
        "3": "src/models/sgd/sgd_model.pkl",
        "4": "src/models/hist_gb/hist_gb_model.pkl",
        "5": "src/models/random_trees_embedding/random_trees_embedding_model.pkl",
    }
    model_file = model_files.get(choice)
    return os.path.exists(model_file)


def model_script(choice):
    # Map choices to corresponding model scripts
    model_scripts = {
        "1": "src/models/adaboosting/adaboosting.py",
        "2": "src/models/voting/voting.py",
        "3": "src/models/sgd/sgd.py",
        "4": "src/models/hist_gb/hist_gb.py",
        "5": "src/models/random_trees_embedding/random_trees_embedding.py",
    }
    return model_scripts.get(choice)
