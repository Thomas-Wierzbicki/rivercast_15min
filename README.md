# Rivercast – 15-Minuten Hydrologie-Vorhersage

Rivercast ist ein Beispiel-Projekt zur **Vorhersage von Wasserstand und Durchfluss** im 15-Minuten-Raster.  
Es kombiniert ETL-Pipeline, Feature-Engineering, Machine-Learning (LightGBM Quantile Regression) und optionales Serving via **MQTT / JSON**.  

Das System liefert nicht nur **eine einzelne Prognose**, sondern **Wahrscheinlichkeitsintervalle** (`p10`, `p50`, `p90`) für unterschiedliche Horizonte (z. B. 6h, 12h, 24h).

---

## 📂 Projektstruktur

```
data/
  raw/         # Rohdaten (CSV), z. B. Messungen, ICON, Bodenfeuchte
  interim/     # Zwischenergebnisse (Parquet)
  processed/   # QC-Daten und Features
artifacts/     # Modelle und Vorhersagen
src/
  etl/         # Laden & Zusammenführen der Rohdaten
  features/    # Feature Engineering
  models/      # Training (Quantil-Regression)
  serve/       # Serving (JSON/MQTT)
  utils/       # Metriken (NSE, KGE, RMSE …)
config/        # YAML-Konfigurationsdatei
nodered/       # Beispiel-Flow für Dashboard
```

---

## ⚙️ Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install --upgrade pip -r requirements.txt
```

---

## 🚀 Pipeline ausführen

### Schritt für Schritt

```bash
# 1) ETL: Daten laden (inkl. ICON & Bodenfeuchte) und auf 15-min Raster bringen
python -m src.etl.load

# 2) QC: Plausibilitätsprüfung & Lückenfüllung
python -m src.etl.qc

# 3) Features: Lags, Rollen, Saisonvariablen berechnen
python -m src.features.build_features

# 4) Training: Modelle für alle Horizonte (6h, 12h, 24h) erzeugen
python -m src.models.train_baseline

# 5) Serving: aktuelle Vorhersagen erzeugen (JSON, optional MQTT)
python -m src.serve.publish
```

### Makefile (Kurzform)

```bash
make all
```

---

## 🔮 Vorhersagen

Die Modelle liefern drei **Quantile** pro Horizont:

- **p10** = untere Grenze (optimistisches Szenario)  
- **p50** = Median (zentrale Erwartung)  
- **p90** = obere Grenze (pessimistisches Szenario)  

Beispielauszug aus `artifacts/forecast_latest.json`:

```json
{
  "ts": "2025-09-25T12:00:00Z",
  "station_id": "ERFT_A",
  "target": "h_cm",
  "horizon_steps": 48,
  "horizon_minutes": 720,
  "p10": 118.4,
  "p50": 123.7,
  "p90": 129.1
}
```

Interpretation:  
- In 12 Stunden liegt der Wasserstand sehr wahrscheinlich zwischen **118–129 cm**.  
- Erwartungswert (Median) ist **124 cm**.  

---

## 📊 Evaluationsmetriken

Beim Training werden folgende Kennzahlen berechnet:

- **MAE** – Mean Absolute Error  
- **RMSE** – Root Mean Squared Error  
- **NSE** – Nash–Sutcliffe Efficiency (hydrologisch üblich)  
- **KGE** – Kling–Gupta Efficiency  

Ergebnisse je Horizont stehen in `artifacts/report.json`.

---

## 🌐 MQTT-Anbindung

In `config/config.yaml` einstellen:

```yaml
mqtt:
  enabled: true
  host: "localhost"
  port: 1883
  base_topic: "/rivercast"
```

Dann publisht das Serving in Topics wie:

```
/rivercast/{station_id}/forecast/{horizon_minutes}min/{target}
```

---

## 📈 Node-RED Dashboard

- Flow: `nodered/rivercast_flow.json`  
- Import in Node-RED → MQTT-Broker konfigurieren → Dashboard → Tab **Rivercast**  
- Visualisiert p50 als Linie und Gauge; p10/p90 können als Unsicherheitsband ergänzt werden.

---

## 🗂️ Erweiterungen

- Zielvariable umschaltbar: `target_col: "h_cm"` oder `"q_cms"`.  
- Mehrere Horizonte definierbar: `horizon_steps_list: [24,48,96]` (6h, 12h, 24h).  
- Exogene Features integrierbar: DWD-ICON (Regen, Temperatur), Bodenfeuchte.  
- Einfach in Timeseries-DB (InfluxDB, Timescale) speicherbar → Grafana-Dashboards.  

---

## 🔀 Trainingsmodus

In `config/config.yaml`:

```yaml
training_mode: 'global'       # eine Modellfamilie für alle Stationen
# training_mode: 'per_station'  # separat je Station
```

- **global**: je Horizont ein Modellset unter `artifacts/{H}/...`  
- **per_station**: je Station und Horizont eigenes Modellset unter `artifacts/{station}/{H}/...`  

---

## 📊 Hindcast & Visualisierung

### Hindcast erzeugen
```bash
python -m src.eval.hindcast --horizons 24 48 96 --out data/processed/hindcast
```

### Einzelfall-Fan-Chart
```bash
python -m src.plots.plot_fanchart --station ERFT_001 --horizon 48     --hindcast data/processed/hindcast --out artifacts/plots
```

### Batch-Rendering aller Fan-Charts
```bash
python -m src.plots.batch_render --hindcast data/processed/hindcast --out artifacts/plots
```

---

## 📈 Rollende Skill-Metriken

```bash
python -m src.eval.rolling_metrics --hindcast data/processed/hindcast --window 96     --out data/processed/metrics --plot-out artifacts/metrics_plots
```
- **window=96** → 24 h bei 15-min Raster  
- Outputs: CSV/Parquet + optional PNGs  

---

## ⚡ Live-Dashboards

### Node-RED
- MQTT-Empfang mit Chart + Gauge  
- REST-Endpoint `/rivercast/forecast` (via Node-RED Flow) gibt `forecast_latest.json` im **Grafana-kompatiblen JSON-Format** zurück.  

### Grafana (ohne DB)
- **JSON API Plugin** installieren  
- Datasource URL auf Node-RED Endpoint setzen, z. B.:  
  ```
  http://localhost:1880/rivercast/forecast?station=ERFT_001&horizon=720&target=h_cm&quantiles=p10,p50,p90
  ```
- Timeseries-Panel erstellen → Grafana zeigt automatisch p10/p50/p90 an.  
- Optional: weitere Panels für Hindcast- oder Metrik-Dateien (via CSV-Datasource oder zusätzliche REST-Endpoints).  

---

## ⚠️ Hinweis

Dies ist ein **Demoprojekt** mit synthetischen Daten.  
Für reale hydrologische Anwendungen sind zusätzliche Qualitätskontrollen, Modellvalidierungen und eine enge Abstimmung mit Fachinstitutionen erforderlich.
