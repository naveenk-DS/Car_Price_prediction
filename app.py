import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load model & features
# -------------------------------
model_path = r"E:\Car_Price_Prediction\Car_Price_prediction\model.pkl"
features_path = r"E:\Car_Price_Prediction\Car_Price_prediction\model_features.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(features_path, "rb") as f:
    feature_columns = pickle.load(f)  # exact column names from training

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")

# -------------------------------
# UI
# -------------------------------
st.title("🚗 Car Price Prediction App")
st.write("Enter the car details below to estimate its resale price.")

# Input fields — match order in `feature_columns`
inputs = {}
for col in feature_columns:
    if col == "Mileage_km":
        inputs[col] = st.number_input("Mileage (in kilometers driven)", min_value=0, max_value=500000, value=50000)
    elif col == "Number ownwer":
        inputs[col] = st.selectbox("Number of Previous Owners", [0, 1, 2, 3, 4])
    elif col == "Mileage":
        inputs[col] = st.number_input("Mileage (kmpl)", min_value=0, max_value=50, value=20)
    elif col == "Engine":
        inputs[col] = st.number_input("Engine Capacity (CC)", min_value=500, max_value=5000, value=1200)
    elif col == "Max Power":
        inputs[col] = st.number_input("Max Power (BHP)", min_value=10, max_value=500, value=90)
    elif col == "Torque":
        inputs[col] = st.number_input("Torque (Nm)", min_value=50, max_value=1000, value=200)
    elif col == "Seats":
        inputs[col] = st.number_input("Number of Seats", min_value=2, max_value=10, value=5)
    elif col == "Age of car":
        inputs[col] = st.number_input("Age of Car (years)", min_value=0, max_value=50, value=5)
    elif col == "Body_Type_numeric":
        inputs[col] = st.selectbox("Body Type (Encoded)", [0, 1, 2, 3])
    elif col == "Fuel_Type_numeric":
        inputs[col] = st.selectbox("Fuel Type (Encoded)", [0, 1, 2, 3])
    elif col == "Transmission_numeric":
        inputs[col] = st.selectbox("Transmission Type (Encoded)", [0, 1])

# -------------------------------
# Predict
# -------------------------------
if st.button("Predict Price"):
    try:
        input_df = pd.DataFrame([[inputs[col] for col in feature_columns]], columns=feature_columns)
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Estimated Price: ₹ {prediction:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")

st.markdown("---")
st.caption("Developed by Naveen 🚀")
