import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

np.random.seed(42)
n = 100000

df = pd.DataFrame({
    'distance_km': np.round(np.random.uniform(0.5, 30, n), 2),
    'hour': np.random.randint(0, 24, n),
    'day_of_week': np.random.randint(0, 7, n),
    'is_weekend': np.random.choice([0,1], n, p=[0.7,0.3]),
    'is_rush_hour': np.random.choice([0,1], n),
    'rainfall': np.random.choice([0,0,0,0,1], n),
    'temperature': np.round(np.random.normal(28,6,n),1),
    'is_holiday': np.random.choice([0,1], n, p=[0.9,0.1]),
    'vehicle_encoded': np.random.choice([0,1,2], n),
    'zone_encoded': np.random.choice([0,1,2,3,4,5], n),
})

features = ['distance_km','hour','day_of_week','is_weekend','is_rush_hour',
            'rainfall','temperature','is_holiday','vehicle_encoded','zone_encoded']

eta_target  = (df['distance_km']*3.5 + df['is_rush_hour']*6 +
               df['rainfall']*4 + np.random.normal(0,2,n)).clip(2)
fare_target = (30 + df['distance_km']*12 + df['vehicle_encoded']*12 +
               df['is_rush_hour']*15 + df['rainfall']*20 +
               np.random.normal(0,8,n)).clip(20)

X = df[features]

# ETA model
X_train, X_test, y_train, y_test = train_test_split(X, eta_target, test_size=0.2, random_state=42)
eta_model = RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42, n_jobs=-1)
eta_model.fit(X_train, y_train)
joblib.dump(eta_model, 'eta_model.pkl', compress=3)
print("ETA model saved!")

# Fare model
X_train, X_test, y_train, y_test = train_test_split(X, fare_target, test_size=0.2, random_state=42)
fare_model = RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42, n_jobs=-1)
fare_model.fit(X_train, y_train)
joblib.dump(fare_model, 'fare_model.pkl', compress=3)
print("Fare model saved!")

print("Done! Check file sizes now.")