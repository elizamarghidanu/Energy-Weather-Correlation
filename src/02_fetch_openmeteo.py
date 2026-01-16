import requests
import pandas as pd

LAT, LON = 44.4268, 26.1025
START = "2019-01-01"
END   = "2023-12-31"
TIMEZONE = "Europe/Bucharest"

def main():
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START,
        "end_date": END,
        "hourly": "temperature_2m,precipitation,windspeed_10m,relative_humidity_2m",
        "timezone": TIMEZONE
    }

    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "temp_c": data["hourly"]["temperature_2m"],
        "precip_mm": data["hourly"]["precipitation"],
        "wind_ms": data["hourly"]["windspeed_10m"],
        "rh_pct": data["hourly"]["relative_humidity_2m"],
    })

    out = "data/raw/openmeteo/openmeteo_bucharest_hourly.parquet"
    df.to_parquet(out, index=False)
    print("Saved:", out, "rows=", len(df))
    print(df.head(3))
    print(df.tail(3))

if __name__ == "__main__":
    main()
