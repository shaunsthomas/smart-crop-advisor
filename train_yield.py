import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import joblib
print("1. Loading Crop Yield dataset...")
df = pd.read_csv("data/crop_yield.csv")
df.columns = df.columns.str.strip()
df = df.dropna(subset=['Crop', 'Season', 'State', 'Area', 'Annual_Rainfall', 'Yield'])
df['Crop'] = df['Crop'].astype(str).str.strip().str.title()
df['Season'] = df['Season'].astype(str).str.strip().str.title()
df['State'] = df['State'].astype(str).str.strip().str.title()
if 'Fertilizer' in df.columns:
    df['Fertilizer'] = df['Fertilizer'].fillna(df['Fertilizer'].median())
if 'Pesticide' in df.columns:
    df['Pesticide'] = df['Pesticide'].fillna(df['Pesticide'].median())
yield_cap = df['Yield'].quantile(0.995)
df = df[df['Yield'] <= yield_cap]
print(f"   Cleaned records: {len(df)} across {df['State'].nunique()} Indian states.")
crop_encoder = LabelEncoder()
season_encoder = LabelEncoder()
state_encoder = LabelEncoder()

df['Crop_Code'] = crop_encoder.fit_transform(df['Crop'])
df['Season_Code'] = season_encoder.fit_transform(df['Season'])
df['State_Code'] = state_encoder.fit_transform(df['State'])
features = [
    'Crop_Code', 'Season_Code', 'State_Code',
    'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide'
]
X = df[features]
y = df['Yield']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\n2. Training Random Forest Regressor (60 Trees)...")
model = RandomForestRegressor(
    n_estimators=60,
    max_depth=16,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n3. --> Model R² Score: {r2:.4f} ({r2 * 100:.2f}% variance explained)")
print(f"   --> Average Prediction Error (RMSE): {rmse:.2f} Tonnes/Hectare")
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/yield_model.pkl")

encoders = {
    'crop_encoder': crop_encoder,
    'season_encoder': season_encoder,
    'state_encoder': state_encoder,
    'crops': sorted(list(crop_encoder.classes_)),
    'seasons': sorted(list(season_encoder.classes_)),
    'states': sorted(list(state_encoder.classes_))
}
joblib.dump(encoders, "models/yield_encoders.pkl")

print("\n4. Saved model to: models/yield_model.pkl")
print("   Saved encoders to: models/yield_encoders.pkl")
