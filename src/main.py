from patterns.singleton.ConfigurationManager import ConfigurationManager
from patterns.command.command_pattern import (
    PreprocessCommand,
    Invoker,
    RunClassifierCommand,
)
from patterns.factory.ClassifierFactory import ClassifierFactory
from patterns.observers.ExecutionLogger import ExecutionLogger
from patterns.observers.ResultsDisplayer import ResultsDisplayer
from patterns.observers.StatisticsTracker import StatisticsTracker


# Main function to run the program
def main():
    # Set up our Command Invoker
    invoker = Invoker()

    # Initialize observers
    execution_logger = ExecutionLogger()
    results_displayer = ResultsDisplayer()
    statistics_tracker = StatisticsTracker()

    # Adding preprocessing command to the invoker
    preprocess_command = PreprocessCommand()
    preprocess_command.add_observer(execution_logger)  # Log preprocessing events
    preprocess_command.add_observer(results_displayer)  # Display preprocessing progress
    invoker.add_command(preprocess_command)
    invoker.execute_commands()

    # Ask the user to choose a model
    print(
        "Choose a model to classify your emails.\n"
        "If you haven't already, replace the Email.csv file with your own data.\n"
        "Note: the csv file must be in the same format and renamed to Email.csv."
    )
    print("1. Adaboosting")
    print("2. Voting")
    print("3. SGD")
    print("4. Hist Gradient Boosting")
    print("5. Random Trees Embedding")

    valid_choices = {"1", "2", "3", "4", "5"}
    choice = None

    # Handle invalid input
    while choice not in valid_choices:
        choice = input("Enter 1, 2, 3, 4, or 5: ").strip()
        if choice not in valid_choices:
            print("Invalid input. Please enter a number between 1 and 5.")

    # Adding preprocessing command to the invoker and executing it
    invoker.add_command(PreprocessCommand())
    invoker.execute_commands()
    preprocess_command.add_observer(execution_logger)

    # Path to the preprocessed email CSV
    email_csv_path = "data/Emails_preprocessed.csv"

    # Adding run classifier command to the invoker
    run_classifier_command = RunClassifierCommand(email_csv_path, choice)
    run_classifier_command.add_observer(execution_logger)  # Log classification events
    run_classifier_command.add_observer(results_displayer)  # Display classification results
    run_classifier_command.add_observer(statistics_tracker)  # Track classification statistics
    invoker.add_command(run_classifier_command)
    invoker.execute_commands()  # Execute classification

    # Output classification results
    predictions = run_classifier_command.get_results()
    print("Classification results:")
    for idx, prediction in enumerate(predictions, start=1):
        print(f"Email {idx}: {prediction}")

    # Finished running all tasks
    print("Finished running all tasks")

    # Display classification statistics
    statistics_tracker.display_stats()


if __name__ == "__main__":
    main()
