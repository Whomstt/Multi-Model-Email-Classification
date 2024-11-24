import pandas as pd
import numpy as np
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib
import os


def load_data(file_paths):
    dataframes = [pd.read_csv(file) for file in file_paths]
    combined_df = pd.concat(dataframes, axis=0, ignore_index=True)
    X = combined_df.iloc[:, :-1].values
    y = combined_df.iloc[:, -1].values
    return X, y

#ML Pipeline with random trees embedding
def create_random_trees_embedding_pipeline(n_estimators=100, max_depth=5):
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('random_trees_embedding', RandomTreesEmbedding(
            n_estimators=n_estimators,
            max_depth=max_depth
        )),
        ('classifier', LogisticRegression(
            max_iter=1000,
            solver='liblinear'
        ))
    ])
    return pipeline

def train_and_evaluate_model(X_train, X_test, y_train, y_test, pipeline):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    #trains model to predict y values
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return pipeline

def main():
    file_paths = [
        "data/AppGallery_preprocessed.csv",
        "data/Purchasing_preprocessed.csv"
    ]

    print("Loading data from files...")
    X, y = load_data(file_paths)

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y #20% of data is used for testing
    )

    print("Creating Random Trees Embedding pipeline...")
    random_trees_pipeline = create_random_trees_embedding_pipeline(
        n_estimators=100,
        max_depth=5 #tree depth
    )

    print("Training and evaluating the model...")
    trained_model = train_and_evaluate_model(
        X_train, X_test, y_train, y_test, random_trees_pipeline
    )

    # Saves the model so it can be called later
    model_path = os.path.join("src","models", "random_trees_embedding", "random_trees_embedding_model.pkl")  
    print(f"Saving the model to {model_path}...")
    joblib.dump(trained_model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()