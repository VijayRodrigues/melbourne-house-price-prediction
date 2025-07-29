import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ------------------------------
# Load trained model and features
# ------------------------------
model_path = r"C:\Users\vijay\Downloads\random_forest_model.pkl"
features_path = r"C:\Users\vijay\Downloads\model_features.pkl"

# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load features
with open(features_path, "rb") as f:
    model_features = pickle.load(f)

# ------------------------------
# ------------------------------

st.title("🏡 Melbourne (Australia) House Price Prediction")
st.write("Enter the property details to predict price:")

# User inputs for numeric features
rooms = st.number_input("Rooms", min_value=1, max_value=10, value=3)
bedroom2 = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
bathroom = st.number_input("Bathrooms", min_value=0, max_value=5, value=2)
car = st.number_input("Car Spaces", min_value=0, max_value=10, value=1)
land_size = st.number_input("Land Size (m²)", min_value=0.0, value=500.0)
building_area = st.number_input("Building Area (m²)", min_value=0.0, value=200.0)
distance = st.number_input("Distance from CBD (km)", min_value=0.0, value=10.0)
property_age = st.number_input("Property Age (Years)", min_value=0, value=10)
price_per_room = st.number_input("Price per Room", min_value=0.0, value=0.0)
land_per_room = st.number_input("Land per Room", min_value=0.0, value=0.0)
latitude = st.number_input("Latitude", value=-37.81)
longitude = st.number_input("Longitude", value=144.96)

# ------------------------------
# Dropdowns list for categorical features (One-hot encoded)
# ------------------------------
suburb = st.selectbox("Suburb", ["Brunswick", "Carlton", "Richmond"])
method = st.selectbox("Sale Method", ["PI - property passed in", "S - property sold", "SA - sold after auction", "SP - property sold prior", "VB - vendor bid"])
council_area = st.selectbox("Council Area", [
    "Melbourne City Council",
    "Moreland City Council",
    "Yarra City Council"
])



# ------------------------------
# Prepare the input data
# ------------------------------
input_data = pd.DataFrame([[0] * len(model_features)], columns=model_features)

# Fill numeric features
input_data.at[0, "Rooms"] = rooms
input_data.at[0, "Bedroom2"] = bedroom2
input_data.at[0, "Bathroom"] = bathroom
input_data.at[0, "Car"] = car
input_data.at[0, "Landsize"] = land_size
input_data.at[0, "BuildingArea"] = building_area
input_data.at[0, "Distance"] = distance
input_data.at[0, "PropertyAge"] = property_age
input_data.at[0, "PricePerRoom"] = price_per_room
input_data.at[0, "LandPerRoom"] = land_per_room
input_data.at[0, "Latitude"] = latitude
input_data.at[0, "Longitude"] = longitude

# One-hot encode categorical features
if f"Suburb_{suburb}" in input_data.columns:
    input_data.at[0, f"Suburb_{suburb}"] = 1

if f"Method_{method}" in input_data.columns:
    input_data.at[0, f"Method_{method}"] = 1

if f"CouncilArea_{council_area}" in input_data.columns:
    input_data.at[0, f"CouncilArea_{council_area}"] = 1

# ------------------------------
# Predict the price from model
# ------------------------------
if st.button("Predict Price"):
    predicted_price = model.predict(input_data)[0]
    st.success(f"🏠 Predicted Price: ${predicted_price:,.2f}")
