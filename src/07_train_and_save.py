import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

FEATURES = [
    "temp_c_mean","temp_c_min","temp_c_max",
    "precip_mm_sum","wind_ms_mean","rh_pct_mean",
    "weekday","month","is_weekend"
]
TARGET = "load_mw_daily_mean"

def main():
    df = pd.read_parquet("data/final/dataset_daily.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=FEATURES + [TARGET, "year"]).copy()

    last_year = df["year"].max()
    train = df[df["year"] < last_year].copy()
    test  = df[df["year"] == last_year].copy()

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test,  y_test  = test[FEATURES],  test[TARGET]

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    payload = {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "train_years": sorted(train["year"].unique().tolist()),
        "test_year": int(last_year),
    }
    joblib.dump(payload, "models/rf_model.joblib")
    print("Saved model -> models/rf_model.joblib")
    print("Train years:", payload["train_years"], "Test year:", payload["test_year"])
    print("Train rows:", len(train), "Test rows:", len(test))

if __name__ == "__main__":
    main()
