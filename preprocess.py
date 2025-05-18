import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess(path="data/house_data.csv"):
    df = pd.read_csv(path)

    # Select important numeric features
    features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt", "SalePrice"]
    df = df[features]

    # Drop rows with missing values (for simplicity)
    df = df.dropna()

    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
