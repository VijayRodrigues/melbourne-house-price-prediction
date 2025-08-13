import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date
import tempfile, urllib.request
from pathlib import Path

# Load full pipeline (preprocess + model)

MODEL_URL = "https://raw.githubusercontent.com/<user>/<repo>/<branch>/random_forest_model.pkl"

with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir) / "random_forest_model.pkl"
    urllib.request.urlretrieve(MODEL_URL, tmp_path)
    pipe = joblib.load(tmp_path)

st.title("🏡 Melbourne (Australia) House Price Prediction")
st.write("Enter property details to predict price:")


rooms         = st.number_input("Rooms", min_value=1, max_value=10, value=3)
bedroom2      = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
bathroom      = st.number_input("Bathrooms", min_value=0, max_value=5, value=2)
car           = st.number_input("Car Spaces", min_value=0, max_value=10, value=1)
land_size     = st.number_input("Land Size (m²)", min_value=0.0, value=500.0)
building_area = st.number_input("Building Area (m²)", min_value=0.0, value=200.0)
distance      = st.number_input("Distance from CBD (km)", min_value=0.0, value=10.0)
latitude      = float(st.number_input("Latitude", value=-37.81))
longitude     = float(st.number_input("Longitude", value=144.96))
postcode      = st.number_input("Postcode", min_value=3000, max_value=3999, value=3056)
propertycount = st.number_input("Propertycount (suburb stock)", min_value=0, value=7000)

# Sale date (to derive Sale_Year / Sale_Month)
sale_dt = st.date_input("Sale Date", value=date(2017, 1, 1))

# Property age & engineered ratios
property_age  = st.number_input("Property Age (Years)", min_value=0, value=10)
price_per_room = st.number_input("Price per Room (if known, else leave 0)", min_value=0.0, value=0.0)
land_per_room  = st.number_input("Land per Room (if known, else leave 0)", min_value=0.0, value=0.0)


# UI — categorical selections (for one-hot passthrough columns)

suburb = st.selectbox("Suburb", ["Brunswick", "Carlton", "Richmond"])
ptype  = st.selectbox("Property Type", ["h (house)", "u (unit/apartment)", "t (townhouse)"])
ptype_code = {"h (house)": "h", "u (unit/apartment)": "u", "t (townhouse)": "t"}[ptype]

method_label = st.selectbox(
    "Sale Method",
    ["PI - property passed in", "S - property sold", "SA - sold after auction",
     "SP - property sold prior", "VB - vendor bid"]
)
method_map = {
    "PI - property passed in": "PI",
    "S - property sold": "S",
    "SA - sold after auction": "SA",
    "SP - property sold prior": "SP",
    "VB - vendor bid": "VB",
}
method = method_map[method_label]

council = st.selectbox(
    "Council Area",
    ["Melbourne City Council", "Moreland City Council", "Yarra City Council"]
)


pre = pipe.named_steps.get("preprocess")
# These are the columns that were passed through (already one-hot at training time)
passthrough_cols = []
num_cols = []
cat_cols = []
if pre is not None:
    for name, transformer, cols in pre.transformers_:
        if name == "passthrough_ohe":
            passthrough_cols = list(cols)  # e.g., Suburb_*, Type_*, Method_*, CouncilArea_*
        elif name == "num":
            num_cols = list(cols)
        elif name == "cat":
            cat_cols = list(cols)

# Build the full expected input column set
expected_cols = set(num_cols) | set(cat_cols) | set(passthrough_cols)


# Create a 1-row DataFrame with ALL expected columns initialized to 0 / NaN

row = pd.Series({c: 0 for c in expected_cols}, dtype="float64")


# Fill raw numeric features if the pipeline expects them

def set_if(col, val):
    if col in row.index:
        row[col] = val

set_if("Rooms", rooms)
set_if("Bedroom2", bedroom2)
set_if("Bathroom", bathroom)
set_if("Car", car)
set_if("Landsize", land_size)
set_if("BuildingArea", building_area)
set_if("Distance", distance)
set_if("Latitude", latitude)
set_if("Longitude", longitude)
set_if("Postcode", postcode)
set_if("Propertycount", propertycount)

# Derived fields used in your training (if expected):
sale_year = int(sale_dt.year)
sale_month = int(sale_dt.month)
year_built = max(sale_year - int(property_age), 0)

set_if("PropertyAge", property_age)
set_if("YearBuilt", year_built)
set_if("Sale_Year", sale_year)
set_if("Sale_Month", sale_month)
set_if("PricePerRoom", price_per_room)
set_if("LandPerRoom", land_per_room)

# DistanceBucket (training likely binned distance; align as best as possible)
def distance_bucket_km(d):
    # Example binning that often matches projects; change if your training used a different map
    if d <= 5:   return 0
    if d <= 10:  return 1
    if d <= 15:  return 2
    if d <= 30:  return 3
    return 4

if "DistanceBucket" in row.index:
    row["DistanceBucket"] = distance_bucket_km(distance)

# IsOldProperty flag (adjust threshold if you used another one during training)
if "IsOldProperty" in row.index:
    row["IsOldProperty"] = 1 if property_age >= 50 else 0


# one-hot passthrough columns to 1 based on selections, if they exist

# Suburb_*
key = f"Suburb_{suburb}"
if key in row.index:
    row[key] = 1.0

# Type_h / Type_u / Type_t
tkey = f"Type_{ptype_code}"
if tkey in row.index:
    row[tkey] = 1.0

# Method_*
mkey = f"Method_{method}"
if mkey in row.index:
    row[mkey] = 1.0

# CouncilArea_*
ckey = f"CouncilArea_{council}"
if ckey in row.index:
    row[ckey] = 1.0


# Convert to DataFrame with columns in any order; pipeline will select what it needs

input_df = pd.DataFrame([row])


# Predict

if st.button("Predict Price"):
    try:
        pred = float(pipe.predict(input_df)[0])
        st.success(f"🏠 Predicted Price: ${pred:,.2f}")
    except Exception as e:
        # Helpful debug info if a column still mismatches
        st.error(f"Prediction failed: {e}")
        st.write("Pipeline expects these columns:", sorted(list(expected_cols)))
        st.write("You provided columns:", list(input_df.columns))


