import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os

# ======== Config ==========
FEATURE_FILE = "data/era5_features.csv"
TARGET_FILE = "data/pv_output.csv"
FEATURES = ["temp_2m", "cc", "q", "wind_speed", "hour", "day_of_year"]
TARGET = "pv_output"
TEST_SIZE = 0.2
RANDOM_STATE = 42
# ===========================

def main():
    if not os.path.exists(FEATURE_FILE) or not os.path.exists(TARGET_FILE):
        raise FileNotFoundError("Input CSV files not found. Please check paths.")

    # Load and merge
    print("Loading data...")
    df_feat = pd.read_csv(FEATURE_FILE, parse_dates=["time"])
    df_target = pd.read_csv(TARGET_FILE, parse_dates=["time"])
    df = pd.merge(df_feat, df_target, on=["time", "field"], how="inner").dropna()

    # Train-test split
    print(f"Training XGBoost on {len(df)} samples...")
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False
    )

    # Train model
    model = XGBRegressor(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n🟢 Evaluation:")
    print(f"MAE: {mae:.2f}")
    print(f"R²:  {r2:.3f}")

    # Optional: Save model
    model.save_model("results/xgboost_model.json")
    print("✅ Model saved to results/xgboost_model.json")

if __name__ == "__main__":
    main()
