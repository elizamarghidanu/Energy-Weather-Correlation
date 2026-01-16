import pandas as pd
from pathlib import Path

path = Path(r"data/raw/entsoe")
files = sorted(path.glob("*.csv"))
if not files:
    raise SystemExit("Nu am găsit niciun .csv în data/raw/entsoe")

f = files[0]
print("Inspecting:", f)

# încercăm mai mulți separatori
seps = [",", ";", "\t", "|"]
best = None

for sep in seps:
    try:
        df = pd.read_csv(f, sep=sep, encoding="utf-8-sig", nrows=5)
        # considerăm valid dacă avem >3 coloane
        if df.shape[1] > 3:
            best = (sep, df)
            break
    except Exception:
        pass

if best is None:
    # fallback: încearcă fără sep specificat
    df = pd.read_csv(f, encoding="utf-8-sig", nrows=5)
    print("Fallback read_csv()")
    print("Columns:", list(df.columns))
    print(df.head())
else:
    sep, df = best
    print("Detected sep:", repr(sep))
    print("Columns:", list(df.columns))
    print(df.head())
