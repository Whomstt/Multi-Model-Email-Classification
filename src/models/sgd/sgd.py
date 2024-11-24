import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import os
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

# Function to train SGD classifier
def train_sgd(X_train, y_train):
    """Train an SGD classifier."""
    # Create the SGD model
    sgd_model = SGDClassifier(
        loss='log_loss',  # Use logistic regression for classification
        max_iter=1000,
        tol=1e-3,
        random_state=42
    )
    # Train the model
    sgd_model.fit(X_train, y_train)
    return sgd_model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance and print metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Main function
def main():
    # Paths to the preprocessed CSV files

    
    file_paths = [
        "data/AppGallery_preprocessed.csv",
        "data/Purchasing_preprocessed.csv",
    ]

    print("Absolute file paths:")
    for file_path in file_paths:
        full_path = os.path.abspath(file_path)
        print(f" - {full_path}")

    # Load the data
    print("Loading data from files...")
    X, y = load_data(file_paths)

    # Handle missing values in the data
    print("Handling missing values...")
    X = handle_missing_values(X)

    # Split the data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train the SGD model
    print("Training SGD classifier...")
    model = train_sgd(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, X_test, y_test)

    # Save the model 
    model_path = os.path.join("src","models", "sgd", "sgd_model.pkl")  
    print(f"Saving the model to {model_path}...")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

# Run the script
if __name__ == "__main__":
    main()
