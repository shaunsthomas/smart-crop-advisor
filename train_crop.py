import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
print("Loading Crop Recommendation dataset...")
df = pd.read_csv("data/Crop_recommendation.csv")
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[features]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Total dataset records: {len(df)}")
print(f"  - Training samples: {len(X_train)} (80%)")
print(f"  - Testing samples:  {len(X_test)} (20%)")
print("\nTraining Random Forest model (100 Decision Trees)...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--> Model Test Accuracy: {accuracy * 100:.2f}%")
os.makedirs("models", exist_ok=True)
model_path = "models/crop_model.pkl"
joblib.dump(model, model_path)
print(f"Successfully saved trained model to: {model_path}")
