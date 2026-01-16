import pandas as pd

df = pd.read_parquet("data/final/dataset_daily.parquet")
print("Rows:", len(df))
print("NaN in y (load_mw_daily_mean):", df["load_mw_daily_mean"].isna().sum())

cols = [
    "load_mw_daily_mean",
    "temp_c_mean","temp_c_min","temp_c_max",
    "precip_mm_sum","wind_ms_mean","rh_pct_mean",
    "weekday","month","is_weekend","year"
]
print("\\nNaN per column:")
print(df[cols].isna().sum().sort_values(ascending=False).head(20))

# arată primele rânduri cu NaN în y
bad = df[df["load_mw_daily_mean"].isna()][["date","load_mw_daily_mean","temp_c_mean","precip_mm_sum","year"]].head(20)
print("\\nExamples where y is NaN:")
print(bad)
