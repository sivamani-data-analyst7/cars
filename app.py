import streamlit as st
import pandas as pd
import pickle as pk
import numpy as np

# Load the trained model
try:
    with open("model.pkl", "rb") as f:
        model = pk.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")

# Streamlit app title
st.title("Car Price Prediction App 🚗💰")

st.write("Enter car details to get the estimated selling price:")

# User inputs
name = st.selectbox("Brand", list(range(1, 32)))
year = st.number_input("Year of Manufacture", min_value=2000, max_value=2025, step=1)
km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, step=1000)
fuel = st.selectbox("Fuel Type", [1, 2, 3, 4])  # 1=Diesel, 2=Petrol, 3=LPG, 4=CNG
seller_type = st.selectbox("Seller Type", [1, 2, 3])  # 1=Individual, 2=Dealer, 3=Trustmark Dealer
transmission = st.selectbox("Transmission", [1, 2])  # 1=Manual, 2=Automatic
owner = st.selectbox("Owner Type", [1, 2, 3, 4, 5])  # 1=First, 2=Second, etc.
mileage = st.number_input("Mileage (km/l)", min_value=5.0, max_value=30.0, step=0.1)
engine = st.number_input("Engine Capacity (cc)", min_value=800, max_value=5000, step=100)
max_power = st.number_input("Max Power (bhp)", min_value=40.0, max_value=500.0, step=5.0)
seats = st.number_input("Number of Seats", min_value=2, max_value=10, step=1)

# Prediction button
if st.button("Predict Price"):
    # Prepare input data
    input_data = np.array([[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display result
    st.success(f"Estimated Selling Price: ₹ {round(prediction[0], 2)}")
