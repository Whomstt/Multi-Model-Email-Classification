import os
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import joblib


# Function to load preprocessed data from multiple files
def load_data(file_paths):
    """Load and combine data from multiple preprocessed CSV files."""
    dataframes = [pd.read_csv(file) for file in file_paths]
    combined_df = pd.concat(dataframes, axis=0, ignore_index=True)
    X = combined_df.iloc[:, :-1].values  # Features (all columns except the last one)
    y = combined_df.iloc[:, -1].values   # Target (last column)
    return X, y

# Function to handle missing values using SimpleImputer
def handle_missing_values(X):
    """Handle missing values in feature data."""
    imputer = SimpleImputer(strategy='mean')  # Fill missing values with the column mean
    X_imputed = imputer.fit_transform(X)
    return X_imputed

# Function to train HistGradientBoosting classifier
def train_hist_gb(X_train, y_train):
    """Train a HistGradientBoostingClassifier."""
    hist_gb_model = HistGradientBoostingClassifier(
        max_iter=1000,       # Number of boosting iterations
        learning_rate=0.5,  # Learning rate
        max_depth=10,       # Maximum tree depth
    )
    # Train the model
    hist_gb_model.fit(X_train, y_train)
    return hist_gb_model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance and print metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def main():
    # Get the directory of the current script
    currentdir = os.path.dirname(os.path.abspath(__file__))

    # Construct paths relative to the script's base directory
    project_root = os.path.abspath(os.path.join(currentdir, "../../.."))  # Navigate back to the project root
    appgallery_path = os.path.join(project_root, "data", "AppGallery_preprocessed.csv")
    purchasing_path = os.path.join(project_root, "data", "Purchasing_preprocessed.csv")

    # Load the data
    print("Loading data from files...")
    X, y = load_data([appgallery_path, purchasing_path])

    # Handle missing values in the data
    print("Handling missing values...")
    X = handle_missing_values(X)

    # Split the data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    # Train the HistGradientBoosting model
    print("Training HistGradientBoosting classifier...")
    model = train_hist_gb(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, X_test, y_test)

    # Save the model 
    model_path = os.path.join("src","models", "hist_gb", "hist_gb_model.pkl")  
    print(f"Saving the model to {model_path}...")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


# Run the script
if __name__ == "__main__":
    main()
