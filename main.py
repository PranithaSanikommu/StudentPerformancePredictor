# main.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load CSV data
df = pd.read_csv('student.csv')

# Step 2: Display the dataset
print("\nDataset:")
print(df)

# Step 3: Prepare input and output
X = df[['study_hours', 'attendance']]  # Input features
y = df['pass']                         # Output label (0 = Fail, 1 = Pass)

# Step 4: Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 6: Predict using test data
y_pred = model.predict(X_test)

# Step 7: Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Step 8: Get new input from user
print("\n--- Student Performance Prediction ---")
try:
    hours = float(input("Enter study hours: "))
    attendance = float(input("Enter attendance percentage: "))
    prediction = model.predict([[hours, attendance]])
    result = "Pass ✅" if prediction[0] == 1 else "Fail ❌"
    print(f"\nPrediction Result: {result}")
except:
    print("Invalid input!")
