import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import joblib
import os

# Function to load preprocessed data from multiple files
def load_data(file_paths):
    """Load and combine data from multiple preprocessed CSV files."""
    dataframes = [pd.read_csv(file) for file in file_paths]
    combined_df = pd.concat(dataframes, axis=0, ignore_index=True)
    X = combined_df.iloc[:, :-1].values  # Features are all columns except the last one
    y = combined_df.iloc[:, -1].values   # Target set to Type 2
    return X, y

# Function to handle missing values 
def handle_missing_values(X):
    imputer = SimpleImputer(strategy='mean')  
    X_imputed = imputer.fit_transform(X)
    return X_imputed

# Function to train AdaBoost classifier
def train_adaboost(X_train, y_train):
    """Train an AdaBoost classifier with a decision tree base estimator."""
    # Define the base estimator
    estimator = DecisionTreeClassifier(max_depth=1)
    # Create the AdaBoost model
    adaboost_model = AdaBoostClassifier(
        estimator=estimator,
        n_estimators=500,  # Number of weak classifiers
        learning_rate=1.5,
    )
    # Train the model
    adaboost_model.fit(X_train, y_train)
    return adaboost_model

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
    # Paths to your preprocessed CSV files
    file_paths = [
        "data/AppGallery_preprocessed.csv",
        "data/Purchasing_preprocessed.csv"
    ]

    # Load the data
    print("Loading data from files...")
    X, y = load_data(file_paths)

    # Handle missing values in the data
    print("Handling missing values...")
    X = handle_missing_values(X)

    # Split the data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    # Train the AdaBoost model
    print("Training AdaBoost classifier...")
    model = train_adaboost(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, X_test, y_test)

    
    
    # Save the model to the 'models/adaboosting' folder
    model_path = os.path.join("src","models", "adaboosting", "adaboost_model.pkl")  
    print(f"Saving the model to {model_path}...")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
