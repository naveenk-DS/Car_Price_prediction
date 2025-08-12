from pyexpat import model
import streamlit as st
import pandas as pd
import pickle

# Load trained model
model_path = r"E:\Car_Price_Prediction\Car_Price_prediction\model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")

# Title
st.title("🚗 Car Price Prediction App")
st.write("Enter the car details to estimate its price.")

# Input fields
Mileage_km = st.number_input("Mileage (in kilometers driven)", min_value=0, max_value=500000, value=50000)
Number_owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3, 4])
Mileage = st.number_input("Mileage (kmpl)", min_value=0, max_value=50, value=20)
Engine = st.number_input("Engine Capacity (CC)", min_value=500, max_value=5000, value=1200)
Max_Power = st.number_input("Max Power (BHP)", min_value=10, max_value=500, value=90)
Torque = st.number_input("Torque (Nm)", min_value=50, max_value=1000, value=200)
Seats = st.number_input("Number of Seats", min_value=2, max_value=10, value=5)
Age_of_car = st.number_input("Age of Car (years)", min_value=0, max_value=50, value=5)
Body_Type_numeric = st.selectbox("Body Type (Encoded)", [0, 1, 2, 3])
Fuel_Type_numeric = st.selectbox("Fuel Type (Encoded)", [0, 1, 2, 3])
Transmission_numeric = st.selectbox("Transmission Type (Encoded)", [0, 1])

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame([[
        Mileage_km,
        Number_owner,  # Fixed column name
        Mileage,
        Engine,
        Max_Power,
        Torque,
        Seats,
        Age_of_car,
        Body_Type_numeric,
        Fuel_Type_numeric,
        Transmission_numeric
    ]], columns=[
        "Mileage_km",
        "Number_owner",  # Corrected from "Number ownwer"
        "Mileage",
        "Engine",
        "Max Power",
        "Torque",
        "Seats",
        "Age of car",
        "Body_Type_numeric",
        "Fuel_Type_numeric",
        "Transmission_numeric"
    ])
    
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Estimated Price: ₹ {prediction:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")

st.markdown("---")
st.caption("Developed by Naveen 🚀")
