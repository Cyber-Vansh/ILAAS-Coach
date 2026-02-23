import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

data_path = "data-cleaned/processed_student_mat.csv"
print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path)

target_col = df.columns[-1]

# WE MUST NOT DROP G1 AND G2! The UI explicitly asks for them!
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Use class_weight='balanced' to ensure 'At Risk' is properly learned since it's a minority class
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, "model/saved_model.pkl")
print("\nModel saved successfully to 'model/saved_model.pkl'")
