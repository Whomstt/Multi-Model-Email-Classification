import sys
from patterns.singleton.ConfigurationManager import ConfigurationManager
from patterns.command.command_pattern import (
    PreprocessCommand,
    Invoker,
    RunClassifierCommand,
)


# Main function
def main():
    config_manager = ConfigurationManager()
    invoker = Invoker()

    invoker.add_command(PreprocessCommand())

    print(
        "Choose a model to run after placing your Email.csv\nin the data folder that you want classified:"
    )
    print("1. Adaboosting")
    print("2. Voting")
    print("3. SGD")
    print("4. Hist Gradient Boosting")
    print("5. Random Trees Embedding")
    
    valid_choices = {"1", "2", "3", "4", "5"}  # Set of valid inputs
    choice = None

    while choice not in valid_choices:
        choice = input("Enter 1, 2, 3, 4, or 5: ").strip()
        if choice not in valid_choices:
            print("Invalid input. Please enter a number between 1 and 5.")

    invoker.add_command(RunClassifierCommand(choice))

    invoker.execute_commands()

    # Finished running the model
    print("Finished running all tasks")


if __name__ == "__main__":
    main()
