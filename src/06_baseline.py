import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def main():
    df = pd.read_parquet('data/final/dataset_daily.parquet')
    df['date'] = pd.to_datetime(df['date'])

    last_year = df['year'].max()
    train = df[df['year'] < last_year].copy()
    test  = df[df['year'] == last_year].copy()

    # Baseline 1: medie pe (month, weekday) din train
    grp = train.groupby(['month','weekday'])['load_mw_daily_mean'].mean()
    pred = test.set_index(['month','weekday']).index.map(grp).to_numpy()

    mae = mean_absolute_error(test['load_mw_daily_mean'], pred)
    rmse = np.sqrt(mean_squared_error(test['load_mw_daily_mean'], pred))

    print(f"Baseline (month+weekday mean): MAE={mae:.2f}, RMSE={rmse:.2f}")

if __name__ == '__main__':
    main()
