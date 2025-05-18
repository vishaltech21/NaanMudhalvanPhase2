import streamlit as st
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏠 House Price Predictor")
st.write("Enter the features of the house to get an estimated price:")

# Sample input fields
feature_names = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
input_data = []

for feature in feature_names:
    val = st.number_input(f"{feature}:", min_value=0.0, value=1.0)
    input_data.append(val)

if st.button("Predict Price"):
    input_array = np.array(input_data).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)[0]
    st.success(f"Estimated House Price: ${prediction:,.2f}")
