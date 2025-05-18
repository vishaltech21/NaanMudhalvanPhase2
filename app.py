import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("House Price Predictor")

# User inputs
OverallQual = st.slider("Overall Quality", 1, 10, 5)
GrLivArea = st.number_input("Ground Living Area (sq ft)", 500, 5000, 1500)
GarageCars = st.slider("Garage Cars", 0, 4, 2)
TotalBsmtSF = st.number_input("Basement SF", 0, 3000, 800)
FullBath = st.slider("Full Bathrooms", 0, 4, 2)
YearBuilt = st.number_input("Year Built", 1900, 2023, 2000)

# Predict
if st.button("Predict Price"):
    features = np.array([[OverallQual, GrLivArea, GarageCars, TotalBsmtSF, FullBath, YearBuilt]])
    features_scaled = scaler.transform(features)
    price = model.predict(features_scaled)[0]
    st.success(f"Estimated House Price: ${price:,.2f}")
