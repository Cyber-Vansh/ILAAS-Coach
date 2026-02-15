import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("data-cleaned/processed_student_mat.csv")

X = df.drop("risk_level", axis=1)
y = df["risk_level"]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(C=1.5, max_iter=2000)
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

joblib.dump(model, "src/student_risk_model.pkl")
joblib.dump(scaler, "src/scaler.pkl")
print("Model and Scaler Saved")
