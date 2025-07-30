
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv('student.csv')
print("\nDataset:")
print(df)
X = df[['study_hours', 'attendance', 'prev_grade', 'internet']]  # Updated input features
y = df['pass']                      
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\n--- Student Performance Prediction ---")
try:
    hours = float(input("Enter study hours: "))
    attendance = float(input("Enter attendance percentage: "))
    prev_grade = float(input("Enter previous grade (0–100): "))
    internet = int(input("Has internet access? (1 = Yes, 0 = No): "))
    
    # Create a 2D list with all 4 features
    prediction = model.predict([[hours, attendance, prev_grade, internet]])
    
    result = "Pass ✅" if prediction[0] == 1 else "Fail ❌"
    print(f"\nPrediction Result: {result}")
except Exception as e:
    print("Invalid input!", e)
