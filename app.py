import streamlit as st
import pandas as pd
import joblib
import pickle

# ===============================
# LOAD MODEL & FEATURES
# ===============================
model_path = r"E:\Car_Price_Prediction\Car_Price_prediction\Hyper_model.joblib"
features_path = r"E:\Car_Price_Prediction\Car_Price_prediction\model_features.pkl"

# Load trained model
Hyper_model = joblib.load(model_path)

# Load feature names
with open(features_path, "rb") as f:
    feature_columns = pickle.load(f)

# ===============================
# STREAMLIT SETTINGS
# ===============================
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")
st.title("🚗 Car Price Prediction App")
st.write("Enter the car details to estimate its price.")

# ===============================
# INPUT FIELDS (Dynamic from feature list)
# ===============================
input_data = {}
for col in feature_columns:
    if "_km" in col.lower() or "km" in col.lower():
        input_data[col] = st.number_input(col, min_value=0, max_value=500000, value=50000)
    elif "price" in col.lower():
        input_data[col] = st.number_input(col, min_value=0, max_value=100, value=10)
    elif "year" in col.lower():
        input_data[col] = st.number_input(col, min_value=1990, max_value=2025, value=2015)
    elif "owner" in col.lower():
        input_data[col] = st.selectbox(col, [0, 1, 2, 3, 4])
    elif "fuel" in col.lower() or "transmission" in col.lower() or "body" in col.lower():
        input_data[col] = st.selectbox(col, [0, 1])  # Encoded categorical
    else:
        input_data[col] = st.number_input(col, min_value=0, value=1)

# ===============================
# PREDICT BUTTON
# ===============================
if st.button("Predict Price"):
    try:
        df_input = pd.DataFrame([[input_data[col] for col in feature_columns]], columns=feature_columns)
        prediction = Hyper_model.predict(df_input)[0]
        st.success(f"💰 Estimated Price: ₹ {prediction:,.2f} Lakhs")
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")

st.markdown("---")
st.caption("Developed by Naveen 🚀")
