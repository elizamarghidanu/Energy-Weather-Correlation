import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = "data/final/dataset_daily.parquet"
MODEL_PATH = "models/rf_model.joblib"

st.set_page_config(page_title="Energy vs Weather Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

df = load_data()

st.title("Dashboard: Consum Energie (RO) vs Factori Climatici")
st.caption("ENTSO-E Power Statistics (load) + Open-Meteo (weather), agregat zilnic 2019–2023.")

# Sidebar controls
page = st.sidebar.radio("Secțiune", ["Overview", "EDA", "Anomalii", "Predicții", "Model Comparison"], index=0)

min_d, max_d = df["date"].min().date(), df["date"].max().date()
d1, d2 = st.sidebar.date_input("Interval analiză", value=(min_d, max_d))
mask = (df["date"].dt.date >= d1) & (df["date"].dt.date <= d2)
view = df.loc[mask].copy()

TARGET = "load_mw_daily_mean"
WEATHER_COLS = ["temp_c_mean","temp_c_min","temp_c_max","precip_mm_sum","wind_ms_mean","rh_pct_mean"]

# -------------------------
# Overview
# -------------------------
if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Zile în interval", f"{len(view)}")
    c2.metric("Consum mediu (MW)", f"{view[TARGET].mean():.1f}")
    c3.metric("Temp medie (°C)", f"{view['temp_c_mean'].mean():.1f}")
    c4.metric("Precip total (mm)", f"{view['precip_mm_sum'].sum():.1f}")

    fig = px.line(view, x="date", y=TARGET, title="Consum zilnic mediu (MW)")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(view, x="date", y="temp_c_mean", title="Temperatura medie zilnică (°C)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(view, x="date", y="precip_mm_sum", title="Precipitații zilnice (mm)")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# EDA
# -------------------------
elif page == "EDA":
    st.subheader("Relații și sezonalitate")
    x_col = st.selectbox("Variabilă meteo (X)", ["temp_c_mean","precip_mm_sum","wind_ms_mean","rh_pct_mean"], index=0)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(view, x=x_col, y=TARGET, trendline="ols", title=f"{x_col} vs consum (MW)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        corr = view[[TARGET] + WEATHER_COLS].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Corelații (Pearson)")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        wd = view.groupby("weekday")[TARGET].mean().reset_index()
        wd["day"] = wd["weekday"].map({0:"L",1:"Ma",2:"Mi",3:"J",4:"V",5:"S",6:"D"})
        fig = px.line(wd, x="day", y=TARGET, markers=True, title="Consum mediu pe zi a săptămânii")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        mo = view.groupby("month")[TARGET].mean().reset_index()
        fig = px.bar(mo, x="month", y=TARGET, title="Consum mediu pe lună")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Anomalii
# -------------------------
elif page == "Anomalii":
    st.subheader("Detecție anomalii (z-score robust pe reziduuri)")
    st.caption("Baseline sezonier: media pe (month, weekday). Anomaliile sunt abateri mari față de baseline.")

    work = view.copy()
    grp = work.groupby(["month","weekday"])[TARGET].mean()
    base = pd.Series(list(zip(work["month"], work["weekday"]))).map(grp).to_numpy()
    work["baseline_mw"] = base
    work["residual"] = work[TARGET] - work["baseline_mw"]

    med = np.nanmedian(work["residual"])
    mad = np.nanmedian(np.abs(work["residual"] - med))
    if mad == 0:
        mad = np.std(work["residual"]) if np.std(work["residual"]) != 0 else 1.0
    work["z_robust"] = (work["residual"] - med) / (1.4826 * mad)

    thr = st.slider("Prag |z| pentru anomalii", min_value=2.0, max_value=6.0, value=3.5, step=0.1)
    anom = work[np.abs(work["z_robust"]) >= thr].copy().sort_values("z_robust", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(work, x="date", y="residual", title="Reziduu (consum - baseline)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(work, x="date", y="z_robust", title="z-score robust al reziduului")
        st.plotly_chart(fig, use_container_width=True)

    st.write(f"Anomalii detectate: {len(anom)}")
    show_cols = ["date", TARGET, "baseline_mw", "residual", "z_robust", "temp_c_mean", "precip_mm_sum", "wind_ms_mean"]
    st.dataframe(anom[show_cols].head(50), use_container_width=True)

# -------------------------
# Predicții
# -------------------------
elif page == "Predicții":
    st.subheader("Predicții (RandomForest salvat)")
    try:
        payload = load_model()
        model = payload["model"]
        FEATURES = payload["features"]
        test_year = payload["test_year"]
    except Exception as e:
        st.error(f"Nu găsesc modelul salvat. Rulează: python src/07_train_and_save.py. Eroare: {e}")
        st.stop()

    st.caption(f"Model: RandomForest. Train years: {payload['train_years']}. Test year: {test_year}.")

    test = df[df["year"] == test_year].copy()
    test = test[(test["date"].dt.date >= d1) & (test["date"].dt.date <= d2)].copy()

    if len(test) == 0:
        st.warning("Intervalul selectat nu conține date din anul de test. Selectează un interval care include anul de test.")
        st.stop()

    X = test[FEATURES]
    y = test[TARGET]
    pred = model.predict(X)

    mae = mean_absolute_error(y, pred)
    r = rmse(y, pred)

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (test)", f"{mae:.2f}")
    c2.metric("RMSE (test)", f"{r:.2f}")
    c3.metric("n (test)", f"{len(test)}")

    plot_df = test[["date", TARGET]].copy()
    plot_df["pred_rf"] = pred
    fig = px.line(plot_df, x="date", y=[TARGET, "pred_rf"], title="Real vs Predicție (RF)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Simulare: ce se întâmplă dacă schimbăm temperatura?")
    st.caption("Ajustează temp_c_mean cu un delta și vezi efectul estimat asupra predicției.")

    delta = st.slider("Delta temperatură (°C) aplicat la temp_c_mean", -10.0, 10.0, 0.0, 0.5)
    X_sim = X.copy()
    X_sim["temp_c_mean"] = X_sim["temp_c_mean"] + delta
    pred_sim = model.predict(X_sim)

    sim_df = plot_df.copy()
    sim_df["pred_rf_temp_shift"] = pred_sim
    fig = px.line(sim_df, x="date", y=["pred_rf", "pred_rf_temp_shift"], title="Predicție RF: original vs temperatură modificată")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Model Comparison
# -------------------------
elif page == "Model Comparison":
    st.subheader("Compararea modelelor")
    st.caption("Baseline vs Linear Regression vs Random Forest, evaluare pe ultimul an (test).")

    try:
        metrics = pd.read_csv("results/model_metrics.csv")
        st.dataframe(metrics, use_container_width=True)

        fig = px.bar(metrics, x="model", y="RMSE", title="RMSE pe modele (mai mic = mai bun)")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.bar(metrics, x="model", y="MAE", title="MAE pe modele (mai mic = mai bun)")
        st.plotly_chart(fig, use_container_width=True)

        best = metrics.sort_values("RMSE").iloc[0]
        st.success(f"Cel mai bun model (după RMSE): {best['model']} | RMSE={best['RMSE']:.2f}, MAE={best['MAE']:.2f}")
    except Exception as e:
        st.error("Nu găsesc results/model_metrics.csv. Rulează: python src/08_model_comparison.py")
        st.write("Eroare:", e)
