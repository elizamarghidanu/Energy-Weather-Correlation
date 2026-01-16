import pandas as pd

ENTSOE_IN = "data/processed/entsoe_ro_hourly.parquet"
METEO_IN  = "data/raw/openmeteo/openmeteo_bucharest_hourly.parquet"
OUT_PARQUET = "data/final/dataset_daily.parquet"
OUT_CSV     = "data/final/dataset_daily.csv"

def main():
    load = pd.read_parquet(ENTSOE_IN)
    load["time"] = pd.to_datetime(load["time"])
    load = load.sort_values("time").set_index("time")

    # Consum zilnic: medie MW
    load_daily = load["load_mw"].resample("D").mean().to_frame("load_mw_daily_mean")

    meteo = pd.read_parquet(METEO_IN)
    meteo["time"] = pd.to_datetime(meteo["time"])
    meteo = meteo.sort_values("time").set_index("time")

    meteo_daily = pd.DataFrame({
        "temp_c_mean": meteo["temp_c"].resample("D").mean(),
        "temp_c_min":  meteo["temp_c"].resample("D").min(),
        "temp_c_max":  meteo["temp_c"].resample("D").max(),
        "precip_mm_sum": meteo["precip_mm"].resample("D").sum(),
        "wind_ms_mean": meteo["wind_ms"].resample("D").mean(),
        "rh_pct_mean": meteo["rh_pct"].resample("D").mean(),
    })

    df = load_daily.join(meteo_daily, how="inner").reset_index().rename(columns={"time": "date"})
    df["weekday"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)

    print("Saved:", OUT_PARQUET, "rows=", len(df))
    print(df.head(3))
    print(df.tail(3))

if __name__ == "__main__":
    main()
