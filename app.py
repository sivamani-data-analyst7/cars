import streamlit as st
import pickle as pk
import numpy as np

# Load the trained model
try:
    with open("model.pkl", "rb") as f:
        model = pk.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")

# Load the encoders
try:
    with open("encoders.pkl", "rb") as f:
        le_name, le_fuel, le_seller_type, le_transmission, le_owner = pk.load(f)
except Exception as e:
    st.error(f"Error loading encoders: {e}")

# Debug: Check if encoders are loaded correctly
st.write("Encoders loaded:", le_name.classes_)

# Streamlit app title
st.title("Car Price Prediction App")

st.write("Enter car details to get the estimated selling price:")

# User inputs
name = st.selectbox("Brand", ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
                             'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
                             'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
                             'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
                             'Ambassador', 'Ashok', 'Isuzu', 'Opel'])
year = st.number_input("Year of Manufacture", min_value=2000, max_value=2025, step=1)
km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, step=1000)
fuel = st.selectbox("Fuel Type", ['Diesel', 'Petrol', 'LPG', 'CNG'])
seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
owner = st.selectbox("Owner Type", ['First Owner', 'Second Owner', 'Third Owner' ,'Fourth & Above Owner', 'Test Drive Car'])
mileage = st.number_input("Mileage (km/l)", min_value=5.0, max_value=30.0, step=0.1)
engine = st.number_input("Engine Capacity (cc)", min_value=800, max_value=5000, step=100)
max_power = st.number_input("Max Power (bhp)", min_value=40.0, max_value=500.0, step=5.0)
seats = st.number_input("Number of Seats", min_value=4, max_value=10, step=1)

# Prediction button
if st.button("Predict Price"):
    try:
        # Prepare input data and ensure it has the correct shape for the model
        input_data = np.array([[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]])

        # Encode categorical features using pre-loaded label encoders
        input_data[0][0] = le_name.transform([input_data[0][0]])  # Brand encoding
        input_data[0][3] = le_fuel.transform([input_data[0][3]])  # Fuel encoding
        input_data[0][4] = le_seller_type.transform([input_data[0][4]])  # Seller type encoding
        input_data[0][5] = le_transmission.transform([input_data[0][5]])  # Transmission encoding
        input_data[0][6] = le_owner.transform([input_data[0][6]])  # Owner encoding

        # Ensure input data is in the correct shape: (1, n_features)
        input_data = input_data.reshape(1, -1)  # Reshape in case it's not in the correct shape

        # Make prediction
        prediction = model.predict(input_data)

        # Display result
        st.success(f"Estimated Selling Price: ₹ {round(prediction[0], 2)}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
