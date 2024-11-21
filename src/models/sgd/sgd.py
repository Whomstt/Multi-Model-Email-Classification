from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import  pandas as pd
import numpy as np
import sys
import os


projectroot = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(projectroot)
from src.preprocessing import X_combined, df_appgallery, df_purchasing

#df_appgallery = pd.read_csv('AppGallery.csv')
#df_purchasing = pd.read_csv('Purchasing.csv')

# Assuming labels are provided in a "Label" column in your datasets
y_appgallery = df_appgallery["Label"].values
y_purchasing = df_purchasing["Label"].values

# Combine labels for the two datasets
y_combined = np.concatenate((y_appgallery, y_purchasing), axis=0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.25, random_state=42)

# Train the SGD Classifier
sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000, random_state=42)
sgd.fit(X_train, y_train)

# Test the Classifier
y_pred = sgd.predict(X_test)

# Evaluate Performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
