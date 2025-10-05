import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('Student_data.csv')

le = LabelEncoder()
df['Internet'] = le.fit_transform(df["Internet"])
df['Passed'] = le.fit_transform(df["Passed"])

features = ['Study Hours', 'Attendance',
            'Past Scores', 'Sleep Hours']

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])

X = df_scaled[features]
y = df_scaled['Passed']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

print("----Predict Your Result----")
try:
    study_hours = float(input("Enter your study hours: "))
    attendance = float(input("Enter your attendance: "))
    past_score = float(input("Enter your past_score: "))
    sleep_hours = float(input("Enter your sleep_hours: "))

    user_input_df = pd.DataFrame([{
        'Study Hours': study_hours,
        'Attendance':  attendance,
        'Past Scores': past_score,
        'Sleep Hours': sleep_hours
    }])

    user_input_scaled = scaler.transform(user_input_df)

    prediction = model.predict(user_input_scaled)[0]

    result = "Pass" if prediction == 1 else "fail"
    print(f"prediction Based on Input: {result}")


except Exception as e:
    print("An error occured", e)

