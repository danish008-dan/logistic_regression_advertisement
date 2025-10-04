# Advertising Data Analysis & Logistic Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
data = pd.read_csv("advertising.csv")

# Quick look of first 5 rows
print(data.head())

# Check missing values
print("\nMissing values before cleaning:")
print(data.isnull().sum())

# Visualize missing values
sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

# Drop irrelevant columns (textual or not useful for ML)
# Keep only numeric + useful categorical features
data.drop(['Ad Topic Line', 'City', 'Country', 'Timestamp'], axis=1, inplace=True)

print("\nData after dropping irrelevant columns:")
print(data.head())

# Prepare features and target
X = data.drop('Clicked on Ad', axis=1)
y = data['Clicked on Ad']

# Train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Logistic Regression
logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train, y_train)

# Predictions
predictions = logmodel.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, predictions))
cm = confusion_matrix(y_test, predictions)
print("\nConfusion matrix:\n", cm)
print("\nClassification report:\n", classification_report(y_test, predictions))

# Confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Example prediction (make sure columns order matches training)
print("\nModel feature columns:", list(X.columns))

example_user_df = pd.DataFrame(
    [[70, 30, 60000, 200, 1]],   # Example: [Daily Time Spent, Age, Area Income, Daily Internet Usage, Male]
    columns=X.columns
)

print("\nExample user:\n", example_user_df)
print("Prediction (0=Not Clicked,1=Clicked):", logmodel.predict(example_user_df))
print("Prediction probabilities [Not Clicked, Clicked]:", logmodel.predict_proba(example_user_df))
