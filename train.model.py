import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

print("--- Starting Local Model Training ---")

# 1. Load the master dataset
DATA_FILE = 'chatbot_master_climate_data.csv'
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"'{DATA_FILE}' not found in D:\\ChatBot. Please ensure it is in this folder.")

df = pd.read_csv(DATA_FILE)

# 2. Select features (inputs) and target (what we want to predict)
features = ['Month', 'temp_batangas', 'rain_station2', 'oni_value']
target = 'rain_batangas'

# Drop missing values to keep training clean
ml_df = df[features + [target]].dropna()

X = ml_df[features]
y = ml_df[target]

# 3. Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate performance
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"✅ Model trained successfully!")
print(f"📊 Mean Absolute Error: {mae:.2f} mm")
print(f"📈 R² Score: {r2:.2f}")

# 6. Save the trained model to disk
joblib.dump(model, 'climate_rain_model.pkl')
print("💾 Model saved locally as 'climate_rain_model.pkl'")