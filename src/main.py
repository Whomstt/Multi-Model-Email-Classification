import sys

print(sys.path)
from patterns.strategy.ClassifierContext import ClassifierContext
from patterns.factory.ClassifierFactory import ClassifierFactory  # Import the factory
from patterns.command.command_pattern import (
    PreprocessCommand,
    Invoker,
    RunClassifierCommand,
)


# Main function
def main():
    invoker = Invoker()

    invoker.add_command(PreprocessCommand())

    print("Choose a model to run:")
    print("1. Adaboosting")
    print("2. Voting")
    print("3. SGD")
    print("4. Hist Gradient Boosting")
    print("5. Random Trees Embedding")
    choice = input("Enter 1, 2, 3, 4, or 5: ").strip()

    invoker.add_command(RunClassifierCommand(choice))

    invoker.execute_commands()

    # Finished running the model
    print("Finished running all tasks")


if __name__ == "__main__":
    main()
