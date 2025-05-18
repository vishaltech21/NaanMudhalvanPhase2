import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Load data
df = pd.read_csv("data/house_data.csv")
print("📌 Raw DataFrame head():")
print(df.head())  # Screenshot: Raw DataFrame

# 2. Check and clean missing values
print("📌 Missing values before cleaning:")
print(df.isnull().sum())  # Screenshot: Missing values

features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt", "SalePrice"]
df = df[features].dropna()

print("✅ Data after dropping missing values:")
print(df.head())  # Screenshot: Cleaned DataFrame

# 3. Before scaling (screenshot)
print("📌 Data before scaling:")
print(df.drop("SalePrice", axis=1).head())  # Screenshot: Before transformation

# 4. Split data
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. After scaling (screenshot)
print("📌 Data after scaling:")
print(pd.DataFrame(X_train_scaled, columns=X.columns).head())  # Screenshot: After transformation

# 7. EDA Visualizations
plt.figure(figsize=(6, 4))
sns.histplot(df['SalePrice'], kde=True)
plt.title("Histogram of SalePrice")
plt.tight_layout()
plt.savefig("histogram_saleprice.png")
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x=df['GrLivArea'])
plt.title("Boxplot of GrLivArea")
plt.tight_layout()
plt.savefig("boxplot_grlivarea.png")
plt.show()

plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("heatmap.png")
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(df['GrLivArea'], df['SalePrice'], alpha=0.5)
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.title("GrLivArea vs SalePrice")
plt.tight_layout()
plt.savefig("scatterplot.png")
plt.show()

# 8. Model training
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("✅ Model training complete!")  # Screenshot: training message

# 9. Model evaluation
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("📈 Evaluation Metrics:")
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")  # Screenshot: evaluation metrics

# 10. Prediction plot
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted Sale Price")
plt.grid(True)
plt.tight_layout()
plt.savefig("prediction_vs_actual.png")
plt.show()

# 11. Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ model.pkl and scaler.pkl saved.")  # Screenshot optional
