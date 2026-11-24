import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Smart Crop, Soil & Fertilizer Advisor", page_icon="🌾")

st.title("🌾 Smart Crop, Soil & Fertilizer Yield Advisor")
st.write("An AI-powered decision support system for farmers (Mini Project Demo).")
st.write("---")

# 2. Load Models and Encoders
@st.cache_resource
def load_all_models():
    crop_m = joblib.load("models/crop_model.pkl")
    fert_m = joblib.load("models/fertilizer_model.pkl")
    fert_enc = joblib.load("models/fertilizer_encoders.pkl")
    yield_m = joblib.load("models/yield_model.pkl")
    yield_enc = joblib.load("models/yield_encoders.pkl")
    return crop_m, fert_m, fert_enc, yield_m, yield_enc

crop_model, fert_model, fert_encoders, yield_model, yield_encoders = load_all_models()

# 3. Input Form
st.subheader("📋 Enter Field & Soil Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🧪 Soil Nutrients**")
    n = st.number_input("Nitrogen (N) kg/ha", min_value=0.0, max_value=200.0, value=90.0)
    p = st.number_input("Phosphorus (P) kg/ha", min_value=0.0, max_value=200.0, value=42.0)
    k = st.number_input("Potassium (K) kg/ha", min_value=0.0, max_value=250.0, value=43.0)
    ph = st.number_input("Soil pH", min_value=3.0, max_value=10.0, value=6.5, step=0.1)

with col2:
    st.markdown("**🌦️ Weather & Soil Type**")
    temp = st.number_input("Temperature (°C)", min_value=5.0, max_value=55.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=10.0, max_value=100.0, value=80.0)
    moisture = st.number_input("Soil Moisture (%)", min_value=5.0, max_value=100.0, value=40.0)
    soil_type = st.selectbox("Soil Type", fert_encoders['soil_encoder'].classes_)

with col3:
    st.markdown("**📍 Location & Field Size**")
    rainfall = st.number_input("Rainfall (mm)", min_value=10.0, max_value=4000.0, value=1100.0)
    state = st.selectbox("State", yield_encoders['states'])
    season = st.selectbox("Season", yield_encoders['seasons'])
    area = st.number_input("Field Area (Hectares)", min_value=0.1, max_value=1000.0, value=2.0)

st.write("---")

# 4. Prediction Button & Result Display
if st.button("🚀 Get Smart Farming Advice", type="primary", use_container_width=True):
    # A. Predict Crop
    crop_input = pd.DataFrame([{
        'N': n, 'P': p, 'K': k,
        'temperature': temp, 'humidity': humidity,
        'ph': ph, 'rainfall': rainfall
    }])
    recommended_crop = crop_model.predict(crop_input)[0].title()

    # B. Predict Fertilizer
    soil_code = fert_encoders['soil_encoder'].transform([soil_type])[0]
    fert_crop_classes = fert_encoders['crop_encoder'].classes_
    target_crop = recommended_crop if recommended_crop in fert_crop_classes else fert_crop_classes[0]
    crop_code = fert_encoders['crop_encoder'].transform([target_crop])[0]
    
    fert_input = pd.DataFrame([{
        'Temparature': temp, 'Humidity': humidity, 'Moisture': moisture,
        'Soil_Code': soil_code, 'Crop_Code': crop_code,
        'Nitrogen': n, 'Potassium': k, 'Phosphorous': p
    }])
    fert_code_pred = fert_model.predict(fert_input)[0]
    recommended_fertilizer = fert_encoders['fertilizer_encoder'].inverse_transform([fert_code_pred])[0]

    # C. Predict Yield
    yield_crop_classes = yield_encoders['crops']
    y_crop = recommended_crop if recommended_crop in yield_crop_classes else yield_crop_classes[0]
    
    y_crop_code = yield_encoders['crop_encoder'].transform([y_crop])[0]
    y_season_code = yield_encoders['season_encoder'].transform([season])[0]
    y_state_code = yield_encoders['state_encoder'].transform([state])[0]
    
    fert_qty = area * 120.0
    pest_qty = area * 0.45
    
    yield_input = pd.DataFrame([{
        'Crop_Code': y_crop_code,
        'Season_Code': y_season_code,
        'State_Code': y_state_code,
        'Area': area,
        'Annual_Rainfall': rainfall,
        'Fertilizer': fert_qty,
        'Pesticide': pest_qty
    }])
    predicted_yield = round(float(yield_model.predict(yield_input)[0]), 2)
    total_harvest = round(predicted_yield * area, 2)

    # D. Display Results
    st.subheader("🎯 Recommendations for Your Field:")
    
    r1, r2, r3 = st.columns(3)
    
    with r1:
        st.success("### 🌾 Best Crop to Grow")
        st.write(f"## **{recommended_crop}**")
        st.caption("Matches your soil NPK & climate.")
        
    with r2:
        st.info("### 🧪 Best Fertilizer")
        st.write(f"## **{recommended_fertilizer}**")
        st.caption(f"Optimized for {soil_type} soil.")
        
    with r3:
        st.warning("### 📈 Expected Harvest Yield")
        st.write(f"## **{predicted_yield} T/Ha**")
        st.caption(f"Total: ~**{total_harvest} Tonnes** for {area} Ha.")
