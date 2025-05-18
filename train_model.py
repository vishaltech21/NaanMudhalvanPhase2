import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Sample dataset
data = {
    'OverallQual': [7, 6, 7, 7, 8],
    'GrLivArea': [1710, 1262, 1786, 1717, 2198],
    'GarageCars': [2, 2, 2, 3, 3],
    'TotalBsmtSF': [856, 1262, 920, 756, 1145],
    'FullBath': [2, 2, 2, 1, 2],
    'YearBuilt': [2003, 1976, 2001, 1915, 2000],
    'SalePrice': [208500, 181500, 223500, 140000, 250000]
}
df = pd.DataFrame(data)

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestRegressor()
model.fit(X_scaled, y)

# Save the model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
