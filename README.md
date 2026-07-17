<div align="center">

# 🌫️ India AQI Intelligence Dashboard

### Interactive Air Quality Analysis & 2026 ML Forecasting

<img src="https://img.shields.io/badge/Domain-Environmental%20Data-2563EB?style=for-the-badge">
<img src="https://img.shields.io/badge/Category-Data%20Visualization-EA580C?style=for-the-badge">
<img src="https://img.shields.io/badge/Type-ML%20Forecasting-16A34A?style=for-the-badge">
<img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge">

> A Python-generated, single-file interactive dashboard analyzing India's air quality data (2022–2025) across 36 states/UTs, with machine-learning forecasts for 2026 — plus a companion methodology report.

</div>

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Files](#files)
- [Pipeline](#pipeline-mainpy)
- [Dashboard Tabs](#dashboard-tabs-aqihtmlhtml)
- [Prediction Methodology](#prediction-methodology)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Data Source](#data-source)

---

## What This Project Does

`main.py` reads a raw AQI dataset, runs the full analysis (national/state/regional trends, pollutant breakdowns, seasonal patterns, ML forecasting), builds 30+ interactive Plotly charts, and writes everything into one self-contained HTML dashboard (`AQIHTML.html`) — no server, no build step, just open the file in a browser.

`methodology.html` is a separate written report explaining the data, the analysis approach, and what each dashboard tab shows.

---

## Files

| File | What It Is |
|---|---|
| `AQI.csv` | Raw dataset — 262,980 records of `country, state, city, last_update, pollutant_id, pollutant_min, pollutant_max, pollutant_avg` |
| `main.py` | The analysis pipeline — loads the CSV, computes every aggregate/chart/prediction, and generates `AQIHTML.html` |
| `AQIHTML.html` | The generated output — a 10-tab interactive dashboard (already built; running `main.py` regenerates it) |
| `methodology.html` | A written FDA-style methodology report describing the dataset and the analysis behind each dashboard tab |

> **Note:** `AQIHTML.html` is approximately 176 MB, which exceeds GitHub's 100 MB hard file-size limit for repository commits — it is therefore not stored directly in this repository. Instead, it is distributed via [GitHub Releases](https://github.com/akshatsingh1427/AQI-Dashboard-India/releases), which supports individual assets up to 2 GB. Download the latest release to view the dashboard directly, or run `python main.py` locally to regenerate it from `AQI.csv`.

---

## Pipeline (`main.py`)

1. **Load & feature-engineer** — parses `last_update` into `year`, `month`, `month_name`, `quarter`
2. **Aggregate** — national yearly trend, per-pollutant averages, monthly pivot, state averages (top 15 most polluted, top 10 cleanest)
3. **Region mapping** — every state/UT is manually mapped into one of 7 regions (North, South, East, West, Central, Northeast, Islands) for regional comparison
4. **Build 20+ base charts** — line, bar, heatmap, box, violin, scatter, pie, sunburst, treemap, waterfall, funnel, radar, area, and an India choropleth map (fetches a public India-states GeoJSON at runtime)
5. **2026 predictions** — `predict_2026()` fits a degree-2 polynomial regression per state/region/nationally on `year → pollutant_avg` (falls back to plain linear regression if that fails), producing a predicted 2026 AQI, % change vs. historical average, and a trend label (Increasing / Decreasing / Stable)
6. **Assemble dashboard** — every chart and prediction table is embedded into one HTML file with a dark, glassmorphism-styled UI, tab navigation, KPI cards, and a chart-description tooltip system
7. **Write output** — saves the finished dashboard to `AQIHTML.html`

---

## Dashboard Tabs (`AQIHTML.html`)

| Tab | Contents |
|---|---|
| **Overview** | National trend + 2026 forecast, pollutant distribution, regional comparison, monthly heatmap, seasonal area chart, quarterly trends |
| **Predictions** | 2026 ML forecasts (national/region/state), historical vs. predicted comparison, waterfall of YoY change, full state & region prediction tables |
| **States** | Top 15 most polluted / top 10 cleanest states, dominant pollutant per state, city-count vs. pollution scatter, state × pollutant heatmap |
| **Regions** | 7-region comparison, year-over-year trends, pollution share pie, pollutant radar profile, box plot variability, top/bottom 3 states per region |
| **Pollutants** | Concentration levels, violin distribution, full state × pollutant matrix, dominant-pollutant table |
| **Advanced** | Sunburst (region → state), treemap, radar, box, waterfall, area, scatter — deeper cuts of the above |
| **Time Series** | Monthly heatmap, quarterly trends, seasonal area chart, YoY waterfall |
| **Rankings** | Funnel of top 10 polluted states, city-count scatter, cleanest/most-polluted rankings |
| **India Map** | Choropleth map of India colored by state AQI (green → yellow → orange → red), with zone counts |
| **Awareness** | AQI scale explainer, month-by-month pollution calendar with causes/precautions, year-round public health guidance, emergency helpline numbers |

---

## Prediction Methodology

```
For each state / region / national level:
  X = years (2022–2025), y = average AQI per year
  Fit PolynomialFeatures(degree=2) + LinearRegression
  If that fails → fall back to plain LinearRegression
  Predict AQI for 2026 (floored at 0)

trend = Increasing / Decreasing / Stable  (predicted vs. historical average)
```

This is documented in more depth, tab by tab, in `methodology.html`.

---

## Tech Stack

<div align="center">

<img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white">
<img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white">
<img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">

</div>

| Layer | Technology |
|---|---|
| **Data Processing** | pandas, numpy |
| **Visualization** | Plotly (Express + Graph Objects) |
| **Modeling** | scikit-learn — `LinearRegression`, `PolynomialFeatures` |
| **Map Data** | Public India-states GeoJSON, fetched via `urllib.request` at runtime |
| **Output** | Single self-contained HTML file (Plotly charts embedded, no external chart dependencies beyond the Plotly CDN script) |
| **Dashboard Styling** | Vanilla CSS — dark theme, glassmorphism, scroll-triggered animations |

---

## How to Run

```bash
pip install pandas plotly numpy scikit-learn

# Keep AQI.csv in the same folder as main.py
python main.py

# Opens/produces AQIHTML.html — open it in any browser
```

> Requires internet access at runtime for the India choropleth map (fetches a GeoJSON from a public gist); if that fetch fails, the dashboard automatically falls back to a bar chart of state-wise AQI instead of the map.

---

## Data Source

`AQI.csv` — India air quality monitoring records (2022–2025) covering pollutant readings (PM2.5, PM10, NO2, SO2, CO, OZONE, NH3) across every state/UT and multiple cities per state, with min/max/average concentration per reading.

<div align="center">

**Built with Python, Plotly, and scikit-learn — one file in, one dashboard out.**

</div>
