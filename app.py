import streamlit as st
import pickle
import numpy as np
import json

# ── Load Models ──
with open('eta_model.pkl', 'rb') as f:
    eta_model = pickle.load(f)

with open('fare_model.pkl', 'rb') as f:
    fare_model = pickle.load(f)

# ── Page Config ──
st.set_page_config(
    page_title="Rapido ETA & Fare Predictor",
    page_icon="🛵",
    layout="centered"
)

# ── Header ──
st.markdown("""
    <h1 style='text-align:center; color:#1B3F7A;'>🛵 Rapido ETA & Fare Predictor</h1>
    <p style='text-align:center; color:gray;'>
        Predict ride ETA and fare using Machine Learning
    </p>
    <hr>
""", unsafe_allow_html=True)

# ── Input Form ──
st.subheader("Enter Ride Details")

col1, col2 = st.columns(2)

with col1:
    distance   = st.slider("Distance (km)", 0.5, 30.0, 5.0, step=0.5)
    vehicle    = st.selectbox("Vehicle Type", ["Bike", "Auto", "Cab"])
    zone       = st.selectbox("Pickup Zone", ["Downtown", "Airport", "Suburbs",
                                               "University", "Mall", "Station"])

with col2:
    hour       = st.slider("Hour of Day", 0, 23, 9)
    rainfall   = st.radio("Weather", ["Clear", "Raining"])
    is_weekend = st.radio("Day Type", ["Weekday", "Weekend"])

is_holiday = st.checkbox("Public Holiday?")

# ── Encode Inputs ──
vehicle_map = {"Bike": 0, "Auto": 1, "Cab": 2}
zone_map    = {"Downtown": 0, "Airport": 1, "Suburbs": 2,
               "University": 3, "Mall": 4, "Station": 5}

is_rush = 1 if (7 <= hour <= 10 or 17 <= hour <= 21) else 0

input_data = np.array([[
    distance,
    hour,
    5 if is_weekend == "Weekend" else 2,
    1 if is_weekend == "Weekend" else 0,
    is_rush,
    1 if rainfall == "Raining" else 0,
    28.0,
    1 if is_holiday else 0,
    vehicle_map[vehicle],
    zone_map[zone]
]])

# ── Predict ──
if st.button("🔍 Predict ETA & Fare", use_container_width=True):

    eta  = eta_model.predict(input_data)[0]
    fare = fare_model.predict(input_data)[0]

    # Surge multiplier
    if is_rush and rainfall == "Raining":
        surge = 2.0
        surge_label = "🔴 High Surge"
    elif is_rush or rainfall == "Raining":
        surge = 1.5
        surge_label = "🟡 Moderate Surge"
    else:
        surge = 1.0
        surge_label = "🟢 Normal"

    final_fare = fare * surge

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Prediction Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("🕐 ETA", f"{eta:.1f} min")
    col2.metric("💰 Base Fare", f"Rs. {fare:.0f}")
    col3.metric("⚡ Surge", f"{surge}x", surge_label)

    st.success(f"✅ Final Fare after Surge: **Rs. {final_fare:.0f}**")

    # Breakdown
    st.markdown("### Fare Breakdown")
    st.markdown(f"""
    | Component | Value |
    |-----------|-------|
    | Base Fare | Rs. {fare:.0f} |
    | Surge Multiplier | {surge}x ({surge_label}) |
    | **Final Fare** | **Rs. {final_fare:.0f}** |
    | Estimated ETA | {eta:.1f} minutes |
    | Vehicle | {vehicle} |
    | Distance | {distance} km |
    """)

# ── Footer ──
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color:gray; font-size:12px;'>
    Built with Random Forest ML Model | MAE: 1.69 min (ETA) | Rs.6.85 (Fare) | R²: 0.99
</p>
""", unsafe_allow_html=True)