# 🌾 Smart Crop, Soil & Fertilizer Yield Advisor

An AI-powered agricultural decision-support system built with **Machine Learning** and **Streamlit**.

## 📌 Features
- **Crop Recommendation:** Predicts the most suitable crop based on soil nutrients ($N, P, K, \text{pH}$) and climate parameters with **99.32% accuracy**.
- **Fertilizer Advisory:** Prescribes the optimal fertilizer based on soil texture and nutrient deficiencies with **95.00% accuracy**.
- **Harvest Yield Prediction:** Estimates harvest output in Tonnes/Hectare across Indian states with an **$R^2$ score of 0.9888**.
- **Interactive Web UI:** Simple, reactive dashboard built with Streamlit.

---

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Scikit-Learn, Pandas, NumPy, Joblib
- **Frontend / UI:** Streamlit
- **Algorithms:** Random Forest Classifier & Random Forest Regressor

---

## 🚀 How to Run Locally

1. **Clone this repository:**
   ```bash
   git clone <repo_url>
   cd "mini project(Final Year)"
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models (Optional - pre-trained models included):**
   ```bash
   python train_crop.py
   python train_fertilizer.py
   python train_yield.py
   ```

4. **Launch the web application:**
   ```bash
   python -m streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.
