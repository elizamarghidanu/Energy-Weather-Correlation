import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

FEATURES = [
    "temp_c_mean","temp_c_min","temp_c_max",
    "precip_mm_sum","wind_ms_mean","rh_pct_mean",
    "weekday","month","is_weekend"
]
TARGET = "load_mw_daily_mean"

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def baseline_month_weekday(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    grp_mw = train.groupby(["month", "weekday"])[TARGET].mean()
    key = list(zip(test["month"], test["weekday"]))
    pred = pd.Series(key).map(grp_mw).to_numpy()

    # fallback month mean -> global mean
    if np.isnan(pred).any():
        grp_m = train.groupby("month")[TARGET].mean()
        pred2 = test["month"].map(grp_m).to_numpy()
        pred = np.where(np.isnan(pred), pred2, pred)

    if np.isnan(pred).any():
        pred = np.where(np.isnan(pred), train[TARGET].mean(), pred)

    return pred

def main():
    df = pd.read_parquet("data/final/dataset_daily.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=FEATURES + [TARGET, "year"]).copy()

    test_year = int(df["year"].max())
    train = df[df["year"] < test_year].copy()
    test  = df[df["year"] == test_year].copy()

    y_test = test[TARGET].to_numpy()
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test = test[FEATURES]

    rows = []

    # Baseline
    pred_base = baseline_month_weekday(train, test)
    rows.append({
        "model": "Baseline(month+weekday mean)",
        "test_year": test_year,
        "MAE": mean_absolute_error(y_test, pred_base),
        "RMSE": rmse(y_test, pred_base),
        "n_test": len(test)
    })

    # Linear Regression
    lr = LinearRegression().fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    rows.append({
        "model": "LinearRegression",
        "test_year": test_year,
        "MAE": mean_absolute_error(y_test, pred_lr),
        "RMSE": rmse(y_test, pred_lr),
        "n_test": len(test)
    })

    # Random Forest
    rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    rows.append({
        "model": "RandomForest",
        "test_year": test_year,
        "MAE": mean_absolute_error(y_test, pred_rf),
        "RMSE": rmse(y_test, pred_rf),
        "n_test": len(test)
    })

    out = pd.DataFrame(rows).sort_values("RMSE")
    out_path = "results/model_metrics.csv"
    out.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print(out)

if __name__ == "__main__":
    main()
