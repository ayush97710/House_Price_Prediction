import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_absolute_error, r2_score  
import joblib

# Dataset लोड करें
df = pd.read_csv("House_price.csv")  
df = df[['area', 'bedrooms', 'bathrooms', 'floors', 'condition', 'price']]  

# Convert Categorical Column
encoder = LabelEncoder()
df["condition"] = encoder.fit_transform(df["condition"])  

# Data Split करें  
X = df.drop("price", axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Model ट्रेन करें
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction और Evaluation
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))  

# Model को Save करें  
joblib.dump(model, "house_price_model.pkl")