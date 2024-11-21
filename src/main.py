import os
import sys
import subprocess


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


# Main function to choose which model to run
def main():
    print("Choose a script to run:")
    print("1. Adaboosting")
    print("2. Voting")
    print("3. SGD")
    print("4. Hist Gradient Boosting")
    print("5. Random Trees Embedding")
    choice = input("Enter 1 or 2: ").strip()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    adaboosting_path = os.path.join(base_dir, "models", "adaboosting", "adaboosting.py")
    voting_path = os.path.join(base_dir, "models", "voting", "voting.py")
    sgd_path = os.path.join(base_dir, "models", "sgd", "sgd.py")
    hist_gb = os.path.join(base_dir, "models", "hist_gb", "hist_gb.py")
    random_trees_embedding = os.path.join(
        base_dir, "models", "random_trees_embedding", "random_trees_embedding.py"
    )

    if choice == "1":
        run_script(adaboosting_path)
    elif choice == "2":
        run_script(voting_path)
    elif choice == "3":
        run_script(sgd_path)
    elif choice == "4":
        run_script(hist_gb)
    elif choice == "5":
        run_script(random_trees_embedding)
    else:
        print("Invalid choice. Pick 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    main()
