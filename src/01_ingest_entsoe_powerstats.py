import glob
import pandas as pd

IN_GLOB = r"data/raw/entsoe/monthly_hourly_load_values_*.csv"
OUT_PATH = r"data/processed/entsoe_ro_hourly.parquet"

def read_entsoe_file(path: str) -> pd.DataFrame:
    # încearcă tab, apoi ;, apoi ,
    for sep in ["\t", ";", ","]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
            # dacă separatorul e corect, ar trebui să avem coloana CountryCode
            if "CountryCode" in df.columns and "DateUTC" in df.columns and "Value" in df.columns:
                return df
        except Exception:
            pass

    # fallback: dacă e o singură coloană cu header cu ';', repară prin split
    df0 = pd.read_csv(path, encoding="utf-8-sig")
    if df0.shape[1] == 1:
        col = df0.columns[0]
        if ";" in col:
            df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
            return df

    raise RuntimeError(f"Nu pot detecta separatorul corect pentru: {path}. Columns={list(df0.columns)}")

def main():
    files = sorted(glob.glob(IN_GLOB))
    if not files:
        raise FileNotFoundError(f"Nu găsesc fișiere: {IN_GLOB}")

    dfs = []
    for f in files:
        df = read_entsoe_file(f)
        df["__file"] = f
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # România
    df = df[df["CountryCode"].astype(str).str.upper() == "RO"].copy()

    # parse datetime (ex: 01-01-2019 00:00 sau 10/08/2021 00:00)
    # încercăm întâi zi-lună-an, apoi zi/lună/an
    dt = pd.to_datetime(df["DateUTC"], format="%d-%m-%Y %H:%M", errors="coerce")
    dt2 = pd.to_datetime(df["DateUTC"], format="%d/%m/%Y %H:%M", errors="coerce")
    df["time"] = dt.fillna(dt2)

    df["load_mw"] = pd.to_numeric(df["Value"], errors="coerce")

    df = df.dropna(subset=["time", "load_mw"]).sort_values("time")

    out = df[["time", "load_mw"]].drop_duplicates()

    # coverage pe ani
    out["year"] = out["time"].dt.year
    print("Hourly rows per year (RO):")
    print(out.groupby("year")["time"].count().sort_index())

    out = out.drop(columns=["year"])
    out.to_parquet(OUT_PATH, index=False)

    print("\nSaved:", OUT_PATH)
    print("Rows:", len(out))
    print("Time range:", out["time"].min(), "->", out["time"].max())

if __name__ == "__main__":
    main()
