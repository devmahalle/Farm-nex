import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Load the training data
print("Loading training data...")
data = pd.read_csv('../Data-processed/crop_recommendation.csv')

# Display basic info about the dataset
print(f"Dataset shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(f"Target variable (label) unique values: {data['label'].unique()}")

# Prepare features and target
X = data[['N', 'P', 'K', 'T', 'H', 'ph', 'rainfall']]
y = data['label']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model
print("Saving the model...")
with open('models/RandomForest_new.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as 'models/RandomForest_new.pkl'")

# Test the model with sample data
print("\nTesting with sample data:")
sample_data = np.array([[50, 40, 30, 25, 70, 6.5, 100]])  # Your test values
prediction = model.predict(sample_data)
print(f"Sample prediction: {prediction[0]}")

print("Model retraining completed successfully!")
