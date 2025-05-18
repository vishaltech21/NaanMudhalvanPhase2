import streamlit as st
import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("House Price Predictor")

# Input fields
OverallQual = st.slider("Overall Quality", 1, 10, 5)
GrLivArea = st.number_input("Above Ground Living Area (sqft)", 500, 5000, 1500)
GarageCars = st.slider("Garage Cars", 0, 4, 2)
TotalBsmtSF = st.number_input("Total Basement SF", 0, 3000, 800)
FullBath = st.slider("Full Bathrooms", 0, 3, 2)
YearBuilt = st.number_input("Year Built", 1900, 2023, 2000)

if st.button("Predict"):
    features = np.array([[OverallQual, GrLivArea, GarageCars, TotalBsmtSF, FullBath, YearBuilt]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")
