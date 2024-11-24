import sys
import os
import numpy as np
from patterns.singleton.ConfigurationManager import ConfigurationManager
from patterns.command.command_pattern import (
    PreprocessCommand,
    Invoker,
    RunClassifierCommand,
)
from patterns.factory.ClassifierFactory import (
    ClassifierFactory,
)  # Importing ClassifierFactory


# main.py
def main():
    # Set up invoker and command pattern
    invoker = Invoker()
    invoker.add_command(PreprocessCommand())

    # Ask the user to choose a model
    print(
        "Choose a the model that you want to classify your emails with.\nIf you haven't already, replace the Email.csv file with your own data.\nNote: the csv file must be in the same format and renamed to Email.csv."
    )
    print("1. Adaboosting")
    print("2. Voting")
    print("3. SGD")
    print("4. Hist Gradient Boosting")
    print("5. Random Trees Embedding")

    valid_choices = {"1", "2", "3", "4", "5"}
    choice = None

    while choice not in valid_choices:
        choice = input("Enter 1, 2, 3, 4, or 5: ").strip()
        if choice not in valid_choices:
            print("Invalid input. Please enter a number between 1 and 5.")

    # Execute preprocessing commands
    invoker.execute_commands()

    # Get the classifier strategy from the factory
    classifier_factory = ClassifierFactory()
    strategy = classifier_factory.get_strategy(choice)

    # Create an instance of RunClassifierCommand
    run_classifier_command = RunClassifierCommand(choice)

    # Path to the preprocessed email CSV
    email_csv_path = "data/Emails_preprocessed.csv"

    # Check if the file exists
    if not os.path.exists(email_csv_path):
        print(
            f"Error: {email_csv_path} not found. Please preprocess your email data first."
        )
        sys.exit(1)

    # Classify emails from the preprocessed CSV
    print(f"Classifying emails from {email_csv_path} using the {choice} model...")
    predictions = run_classifier_command.classify_emails_from_csv(
        email_csv_path, strategy
    )

    # Output the results
    print("Classification results:")
    for idx, prediction in enumerate(predictions, start=1):
        print(f"Email {idx}: {prediction}")

    # Finished running all tasks
    print("Finished running all tasks")


if __name__ == "__main__":
    main()
