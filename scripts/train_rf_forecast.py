import numpy as np
import pandas as pd
import hashlib
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

print("========================================")
print("1. DATA IMPROVEMENT: Generating synthetic product series...")

np.random.seed(42)
N_PRODUCTS = 200
WEEKS = 104 # 2 years of weekly data
data_rows = []

feature_cols = [
    "lag_1", "lag_2", "lag_4", "lag_8",
    "rolling_mean_4", "rolling_std_4",
    "trend", "price_scaled", "product_hash", "product_avg_sales",
    "month", "week_of_year", "quarter", "season", "is_holiday_season" # Keeping temporal features that ml_service.py still prepares!
]

for p in range(N_PRODUCTS):
    pid = f"PRD_{1000 + p}"
    product_hash = int(hashlib.md5(pid.encode()).hexdigest(), 16) % 1000 / 1000.0
    
    price = np.random.uniform(5.0, 150.0)
    price_scaled = price / 100.0
    
    base_vol = np.random.uniform(20.0, 300.0)
    # Different trends
    trend_type = np.random.choice(["increasing", "decreasing", "stable", "volatile"])
    
    history = []
    
    for w in range(WEEKS):
        # Time features
        ts = pd.Timestamp('2023-01-01') + pd.Timedelta(weeks=w)
        month = ts.month
        week = ts.isocalendar()[1]
        quarter = ts.quarter
        season = (quarter - 1) % 4 + 1
        is_holiday = 1 if quarter == 4 else 0
        
        # Base math
        val = base_vol
        
        # Apply trend
        if trend_type == "increasing":
            val += w * np.random.uniform(0.5, 2.0)
        elif trend_type == "decreasing":
            val -= w * np.random.uniform(0.5, 1.5)
            
        # Apply seasonality
        # E.g. peaking in summer or winter 
        season_peak = (product_hash * 4) + 1 # 1 to 5
        season_factor = 20 * np.sin(2 * np.pi * (w + (season_peak*10)) / 52)
        val += season_factor
        
        # Apply volatility/noise
        noise_level = 5 if trend_type != "volatile" else 25
        val += np.random.normal(0, noise_level)
        
        # Spikes
        if np.random.rand() > 0.95:
            val += np.random.uniform(30, 80)
            
        val = max(5.0, round(val, 1))
        history.append(val)
        
    product_avg_sales = np.mean(history)
    
    # Feature extraction loop
    for t in range(12, WEEKS - 1):  # Need at least 8 weeks history for lag_8 + 4 context
        curr_ts = pd.Timestamp('2023-01-01') + pd.Timedelta(weeks=t)
        
        l1 = history[t-1]
        l2 = history[t-2]
        l4 = history[t-4]
        l8 = history[t-8]
        
        rm4 = np.mean(history[t-4:t])
        rs4 = np.std(history[t-4:t])
        trend_val = l1 - l4
        
        row = {
            "pid": pid,
            "target": history[t], # Predict next week
            
            "lag_1": l1,
            "lag_2": l2,
            "lag_4": l4,
            "lag_8": l8,
            "rolling_mean_4": rm4,
            "rolling_std_4": rs4,
            "trend": trend_val,
            "price_scaled": price_scaled,
            "product_hash": product_hash,
            "product_avg_sales": product_avg_sales,
            
            "month": curr_ts.month,
            "week_of_year": curr_ts.isocalendar()[1],
            "quarter": curr_ts.quarter,
            "season": (curr_ts.quarter - 1) % 4 + 1,
            "is_holiday_season": 1 if curr_ts.quarter == 4 else 0
        }
        data_rows.append(row)

df = pd.DataFrame(data_rows)
print(f"Generated {len(df)} training rows.")

X = df[feature_cols]
y = df["target"]

print("========================================")
print("2. MODEL TRAINING: Scikit-Learn RandomForestRegressor")

rf = RandomForestRegressor(
    n_estimators=250,        # User requirement >= 200
    max_depth=12,            # User requirement 8-15
    min_samples_leaf=4,      # User requirement >= 3
    random_state=42,         # Fixed seed
    n_jobs=-1
)

rf.fit(X, y)

print("Training Complete. Scoring model...")
preds = rf.predict(X)
mae = mean_absolute_error(y, preds)
print(f"Mean Absolute Error (In-sample): {mae:.2f}")

print("========================================")
print("3. FEATURE IMPORTANCE CHECK")
importances = rf.feature_importances_
for name, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
    print(f"  {name:20s}: {imp:.4f}")

print("========================================")
print("4. VALIDATION: Forecasting on 5 random products")
test_pids = np.random.choice(df['pid'].unique(), 5, replace=False)

for pid in test_pids:
    p_df = df[df['pid'] == pid].tail(1)
    
    vec = p_df[feature_cols].values[0]
    true_val = p_df['target'].values[0]
    
    pred_val = rf.predict([vec])[0]
    print(f"Product: {pid:10s} | Hash: {vec[8]:.3f} | Lag_1: {vec[0]:6.1f} | Prediction: {pred_val:6.1f}")

print("========================================")
print("5. SAVING MODEL")

payload = {
    "model": rf,
    "feature_cols": feature_cols
}

out_path = os.path.join(os.path.dirname(__file__), "models", "rf_forecast_model.pkl")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "wb") as f:
    pickle.dump(payload, f)

print(f"Successfully saved rf_forecast_model.pkl!")
print(f"Path: {out_path}")
print("========================================")
