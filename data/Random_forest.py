import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

df = pd.read_csv('Heart_Disease_Prediction.csv')  # Adjust path if needed

print("Data shape:", df.shape)
print(df.head())

df['Heart Disease'] = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})

print("Missing values:\n", df.isnull().sum())

X = df.drop('Heart Disease', axis=1)
y = df['Heart Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Optional: Plot confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save the model
with open('heart_disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as heart_disease_model.pkl")
