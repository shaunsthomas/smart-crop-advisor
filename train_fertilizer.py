import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the fertilizer dataset
print("1. Loading Fertilizer dataset...")
df = pd.read_csv("data/Fertilizer Prediction.csv")

# 2. Clean column names (strip trailing spaces like 'Humidity ')
df.columns = df.columns.str.strip()

# 3. Create Encoders to turn text categories into numbers
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
fertilizer_encoder = LabelEncoder()

# Fit and transform the text into numbers
df['Soil_Code'] = soil_encoder.fit_transform(df['Soil Type'].str.strip().str.title())
df['Crop_Code'] = crop_encoder.fit_transform(df['Crop Type'].str.strip().str.title())
df['Fertilizer_Code'] = fertilizer_encoder.fit_transform(df['Fertilizer Name'].str.strip())

print("   Soil Types recognized:", list(soil_encoder.classes_))
print("   Fertilizers recognized:", list(fertilizer_encoder.classes_))

# 4. Separate Inputs (X) and Output Target (y)
features = [
    'Temparature', 'Humidity', 'Moisture',
    'Soil_Code', 'Crop_Code',
    'Nitrogen', 'Potassium', 'Phosphorous'
]
X = df[features]
y = df['Fertilizer_Code']

# 5. Split into 80% Train and 20% Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train the Random Forest Classifier (100 trees)
print("\n2. Training Random Forest model for Fertilizer...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Test and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n3. --> Fertilizer Model Test Accuracy: {accuracy * 100:.2f}%")

# 8. Save BOTH the model and the encoders to the models/ folder
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fertilizer_model.pkl")

# Save encoders so the web app can translate farmer inputs into numbers
encoder_dict = {
    'soil_encoder': soil_encoder,
    'crop_encoder': crop_encoder,
    'fertilizer_encoder': fertilizer_encoder
}
joblib.dump(encoder_dict, "models/fertilizer_encoders.pkl")

print("\n4. Saved model to: models/fertilizer_model.pkl")
print("   Saved encoders to: models/fertilizer_encoders.pkl")
