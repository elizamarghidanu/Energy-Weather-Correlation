import pandas as pd

df = pd.read_parquet("data/final/dataset_daily.parquet")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

missing_y = df[df["load_mw_daily_mean"].isna()][["date","year","month"]]
print("Total rows:", len(df))
print("NaN in y:", len(missing_y))

if len(missing_y) > 0:
    print("\nNaN by year:")
    print(missing_y["year"].value_counts().sort_index())
    print("\nFirst 10 missing dates:")
    print(missing_y.head(10))
    print("\nLast 10 missing dates:")
    print(missing_y.tail(10))

# și un rezumat de coverage pe ani
print("\nAvailable days per year (non-NaN y):")
print(df[~df["load_mw_daily_mean"].isna()].groupby("year")["date"].count())
