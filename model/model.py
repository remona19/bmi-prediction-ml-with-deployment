import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
data = pd.read_csv('bmi.csv')

# Check for and remove duplicates
if data.duplicated().any():
    data.drop_duplicates(keep=False, inplace=True)

# Encode categorical variables
lb = LabelEncoder()
data['Gender'] = lb.fit_transform(data['Gender'])

# Split the data into features (X) and target (y)
X = data.drop('Index', axis=1).values
y = data['Index'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Train a KNeighborsClassifier model
knn_best = KNeighborsClassifier(n_neighbors=6)
knn_best.fit(X_train, y_train)

# Test the model on the test set
y_pred_test = knn_best.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print("Test Set Accuracy:", test_accuracy)

# Save the trained model to a file using pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(knn_best, model_file)

print("Model saved as model.pkl")
