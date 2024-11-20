import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Assuming you have a 'label' column for the classification task
# Example of creating a target label column (if not already present)
# df_appgallery['label'] = df_appgallery['LabelColumn']
# df_purchasing['label'] = df_purchasing['LabelColumn']

# Combine the labels (assuming 'label' is the column name for targets)
y_appgallery = df_appgallery['label']
y_purchasing = df_purchasing['label']

# Preprocess the datasets
X_appgallery = preprocess_dataset(df_appgallery, "Interaction content", "Ticket Summary")
X_purchasing = preprocess_dataset(df_purchasing, "Interaction content", "Ticket Summary")

# Combine the datasets
X_combined = np.concatenate((X_appgallery, X_purchasing), axis=0)
y_combined = np.concatenate((y_appgallery, y_purchasing), axis=0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Initialize AdaBoost with a base DecisionTreeClassifier
adaboost_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42)

# Train the model
adaboost_model.fit(X_train, y_train)

# Predict on the test set
y_pred = adaboost_model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Optionally: save the model using joblib or pickle
import joblib
joblib.dump(adaboost_model, 'adaboost_email_classifier.pkl')
