\# Modelarea și Predicția Consumului de Energie Electrică în Funcție de Factorii Climatici (România)



Acest proiect analizează variația consumului de energie electrică (România) în funcție de condițiile meteo și construiește modele de predicție pe date istorice. Pipeline-ul este offline (fără real-time), iar rezultatele sunt prezentate într-un dashboard interactiv (Streamlit).



\## Date utilizate



1\. \*\*ENTSO-E Power Statistics (offline CSV)\*\*



&nbsp;  \* Consum orar (Load) pe țară, filtrat pentru România (RO)

&nbsp;  \* Perioadă: 2019–2023 (inclusiv)



2\. \*\*Open-Meteo Historical Weather API\*\*



&nbsp;  \* Date orare: temperatură, precipitații, vânt, umiditate relativă

&nbsp;  \* Localizare: România

&nbsp;  \* Agregare zilnică pentru integrare cu consumul



\## Obiective



\* Identificarea relațiilor dintre consum și vreme:



&nbsp; \* sezonalitate (vară vs iarnă)

&nbsp; \* variații pe zi a săptămânii (weekday/weekend)

&nbsp; \* relația temperatură–consum

&nbsp; \* influența precipitațiilor/vântului

&nbsp; \* detectarea anomaliilor (abateri față de comportamentul așteptat)

\* Modelarea consumului pe baza variabilelor climatice și a factorilor de calendar:



&nbsp; \* regresie liniară

&nbsp; \* random forest

&nbsp; \* baseline sezonier (month + weekday)



\## Structura proiectului



```

energy-weather-bd/

├─ data/

│  ├─ raw/

│  │  ├─ entsoe/                 # CSV-uri ENTSO-E (offline)

│  │  └─ openmeteo/              # date brute meteo (parquet/json)

│  ├─ processed/                 # entsoe RO orar (parquet)

│  └─ final/                     # dataset zilnic integrat (parquet/csv)

├─ models/                       # modele salvate (joblib)

├─ results/                      # metrici modele (CSV)

└─ src/

&nbsp;  ├─ 01\_ingest\_entsoe\_powerstats.py

&nbsp;  ├─ 02\_fetch\_openmeteo.py

&nbsp;  ├─ 03\_build\_daily\_dataset.py

&nbsp;  ├─ 04\_model.py

&nbsp;  ├─ 05\_nan\_report.py

&nbsp;  ├─ 07\_train\_and\_save.py

&nbsp;  ├─ 08\_model\_comparison.py

&nbsp;  └─ dashboard.py

```



\## Cerințe



\* Windows + PowerShell

\* Python 3.10+ recomandat

\* Dependențe: `pandas`, `numpy`, `scikit-learn`, `streamlit`, `plotly`, `joblib`



\## Instalare



1\. Creează și activează un virtual environment:



```powershell

python -m venv .venv

.venv\\Scripts\\Activate.ps1

```



2\. Instalează dependențele:



```powershell

pip install -r requirements.txt

```



Dacă nu ai `requirements.txt`, instalează rapid:



```powershell

pip install pandas numpy scikit-learn streamlit plotly joblib -q

```



\## Pregătirea datelor (ENTSO-E offline)



Pune fișierele ENTSO-E în:



```

data/raw/entsoe/

```



Exemplu nume fișiere:



\* `monthly\_hourly\_load\_values\_2019.csv`

\* `monthly\_hourly\_load\_values\_2020.csv`

\* `monthly\_hourly\_load\_values\_2021.csv`

\* `monthly\_hourly\_load\_values\_2022.csv`

\* `monthly\_hourly\_load\_values\_2023.csv`



Notă: unele fișiere pot avea separator `\\t` sau `;`. Scriptul de ingestie detectează automat separatorul.



\## Rulare pipeline (end-to-end)



Din rădăcina repo-ului:



```powershell

python src\\01\_ingest\_entsoe\_powerstats.py

python src\\02\_fetch\_openmeteo.py

python src\\03\_build\_daily\_dataset.py

python src\\04\_model.py

python src\\08\_model\_comparison.py

```



Verificare integritate (fără NaN în target):



```powershell

python src\\05\_nan\_report.py

```



\## Antrenare și salvare model (pentru dashboard)



```powershell

mkdir models -Force

python src\\07\_train\_and\_save.py

```



\## Dashboard (Streamlit)



```powershell

streamlit run src\\dashboard.py

```



Dashboard-ul include:



\* \*\*Overview\*\*: trend consum, temperatură, precipitații

\* \*\*EDA\*\*: scatter + trendline, corelații, sezonalitate

\* \*\*Anomalii\*\*: reziduuri față de baseline (month+weekday) + z-score robust + tabel

\* \*\*Predicții\*\*: Real vs Predicție (RF), MAE/RMSE, simulare “temperature shift”

\* \*\*Model Comparison\*\*: baseline vs LR vs RF (MAE/RMSE) din `results/model\_metrics.csv`



\## Metodologie (pe scurt)



1\. Ingest ENTSO-E offline + filtrare RO

2\. Descărcare meteo (București) via Open-Meteo

3\. Agregare zilnică și integrare pe axa temporală

4\. Analiză exploratorie (trend, sezonalitate, corelații)

5\. Modelare:



&nbsp;  \* Baseline sezonier (month+weekday)

&nbsp;  \* Linear Regression

&nbsp;  \* Random Forest

6\. Evaluare pe ultimul an (test) + interpretare (feature importances)



\## Rezultate (exemplu)



\* Random Forest depășește Linear Regression pe MAE/RMSE (capturat în `results/model\_metrics.csv`).

\* Importanța temperaturii medii este dominantă, urmată de factori de calendar (weekday/weekend).



\## Reproducibilitate



\* Proiectul rulează offline după ce datele ENTSO-E sunt puse în `data/raw/entsoe/`.

\* `data/` poate fi exclus din Git (recomandat prin `.gitignore`), iar pipeline-ul poate fi refăcut pe alt sistem.



\## Licență



Datele rămân sub licențele furnizorilor (ENTSO-E / Open-Meteo). Codul poate fi licențiat separat (ex. MIT) dacă este necesar.



