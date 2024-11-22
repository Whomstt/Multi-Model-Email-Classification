import os
import sys
import subprocess
import sys

print(sys.path)
from patterns.strategy.ClassifierContext import ClassifierContext
from patterns.factory.ClassifierFactory import ClassifierFactory  # Import the factory
import preprocessing
import preprocessing_unseen


# Paths for the preprocessed CSV files
purchasing_file = "data/Purchasing_preprocessed.csv"
appgallery_file = "data/AppGallery_preprocessed.csv"


# Run preprocessing script
def run_preprocessing():
    if not os.path.exists(purchasing_file):
        preprocessing.preprocess_data("data/Purchasing.csv")
    elif not os.path.exists(appgallery_file):
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


# Main function to choose which model to run using the Strategy Pattern
def main():
    # Run preprocessing (includes checking if preprocessed files exist)
    run_preprocessing()

    print("Preprocessed CSV files found. Proceeding with the rest of the program.")
    print("Choose a model to run:")
    print("1. Adaboosting")
    print("2. Voting")
    print("3. SGD")
    print("4. Hist Gradient Boosting")
    print("5. Random Trees Embedding")
    choice = input("Enter 1, 2, 3, 4, or 5: ").strip()

    # Create the strategy based on the user input
    classifier_factory = ClassifierFactory()
    strategy = classifier_factory.get_strategy(choice)

    # Create a ClassifierContext to use the selected strategy
    context = ClassifierContext(strategy)

    # Run the selected model using the context
    context.run_classifier_model()

    # Finished running the model
    print("Finished running the model.")


if __name__ == "__main__":
    main()
