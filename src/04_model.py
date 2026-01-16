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

def main():
    df = pd.read_parquet("data/final/dataset_daily.parquet")

    # Curățare: elimină rândurile unde target sau oricare feature este NaN
    keep_cols = ["year"] + FEATURES + [TARGET]
    before = len(df)
    df = df.dropna(subset=keep_cols).copy()
    after = len(df)
    print(f"Dropped rows with NaN: {before - after} (remaining {after})")

    y = df[TARGET]
    X = df[FEATURES]

    # test = ultimul an
    last_year = df["year"].max()
    train_mask = df["year"] < last_year

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[~train_mask], y[~train_mask]

    lr = LinearRegression().fit(X_train, y_train)
    pred_lr = lr.predict(X_test)

    rf = RandomForestRegressor(n_estimators=400, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    def report(name, pred):
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        print(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}")

    report("LinearRegression", pred_lr)
    report("RandomForest", pred_rf)

    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\\nTop feature importances (RF):")
    print(imp.head(10))

if __name__ == "__main__":
    main()
