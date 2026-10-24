# STAC WARS · PS 2 — Solar Radiation Analysis & Prediction Report
**Mission:** *Voyage-123 Decoded: Read the Sun, Power the Rebellion*  
**Dataset:** `SolarPrediction.csv` (`32,686` observations · `118` active days · `2016-09-01` to `2016-12-31`)

---

## 1. Executive Summary & Dataset Audit
The Alliance probe **Voyage-123** logged `32,686` meteorological and solar irradiance measurements at ~5-minute intervals between **September 1, 2016 (`00:00:08`)** and **December 31, 2016 (`23:55:01`)**.
- **Completeness:** Zero missing (`NaN`) cells across all 11 columns (`UNIXTime`, `Data`, `Time`, `Radiation`, `Temperature`, `Pressure`, `Humidity`, `WindDirection(Degrees)`, `Speed`, `TimeSunRise`, `TimeSunSet`).
- **Unlogged Probe Blackout Days:** Across the 122-day calendar window (Sep–Dec 2016), exactly `118` unique dates were logged; `4` dates (`2016-09-30`, `2016-11-30`, `2016-12-06`, and `2016-12-07`) had telemetry dropouts.
- **Core Metric Distributions:**
  - **Solar Radiation (`W/m²`):** Mean `207.12`, Std `315.92`, Min `1.11`, Median `2.66`, IQR `1.23–354.24`, Max `1,601.26`
  - **Temperature (`°F`):** Mean `51.10`, Std `6.20`, Min `34.00`, Max `71.00`
  - **Humidity (`%`):** Mean `75.02`, Std `25.99`, Min `8.00`, Max `103.00`
  - **Barometric Pressure (`Hg`):** Mean `30.42`, Std `0.05`, Min `30.19`, Max `30.56`
  - **Wind Speed (`mph`):** Mean `6.24`, Std `3.49`, Min `0.00`, Max `40.50`
  - **Daylight Duration (`hours`):** Mean `11.51`, shrinking from `12.52 hrs` (Sep 1) to `10.93 hrs` (Dec 31)

---

## 2. Part A — Data Visualizations
All six high-resolution visualizations are saved in the `plots/` directory and embedded interactively in the presentation website (`index.html`):
1. `plots/01_radiation_timeseries.png` — **24-Hour Diurnal Solar Profile** (Mean, Median, IQR) & **118-Day Daily Max/Average Timeline**.
2. `plots/02_environmental_correlations.png` — **Radiation vs. Temperature (`r = +0.735`)**, **Daytime Radiation vs. Humidity (`r = -0.226`)**, and the **7-Variable Correlation Matrix**.
3. `plots/03_outliers_and_distributions.png` — **Outlier Anatomy of the `2016-09-04` (`1,601.26 W/m²`) Cloud-Edge Lensing Spike** & **Monthly Daytime Boxplots**.
4. `plots/04_bonus_hysteresis_and_heatmap.png` — **Bonus Visualization:** **Diurnal Thermal Hysteresis Phase Loop (`00:00→23:00`)** & **118-Day × 24-Hour Solar Carpet Plot**.
5. `plots/05_part_c_top_days.png` — **Top 5 Average Radiation Days vs. Top 5 Peak Radiation Days**.
6. `plots/06_ml_prediction_evaluation.png` — **Machine Learning Regression Benchmark (`R²` & `RMSE`) and Feature Importance Ranking**.

---

## 3. Part B — Analytical Inferences & Evidence

### B.1 During what time of day is solar radiation generally highest?
Solar radiation peaks sharply during the **midday window (`10:00–14:00`)**, with the absolute diurnal peak occurring during **`12:00–12:59`**:
- **`12:00–12:59`:** Mean = **`727.47 W/m²`**, Median = **`862.46 W/m²`**, Max = **`1,601.26 W/m²`**
- **`11:00–11:59`:** Mean = **`713.70 W/m²`**, Median = **`836.94 W/m²`**, Max = **`1,475.40 W/m²`**
- **`10:00–10:59`:** Mean = **`681.41 W/m²`**, Median = **`772.00 W/m²`**, Max = **`1,368.35 W/m²`**
- **`13:00–13:59`:** Mean = **`649.52 W/m²`**, Median = **`794.75 W/m²`**, Max = **`1,288.17 W/m²`**
- **Nighttime (`19:00–05:59`):** Mean = **`1.32–1.39 W/m²`** (pyranometer thermal zero-offset; minimum `1.11 W/m²`).

### B.2 Which environmental parameter has the strongest relationship with radiation?
**Temperature (`°F`)** exhibits the strongest relationship with solar radiation (**Pearson $r = +0.7350$**). All environmental correlations with `Radiation`:
| Parameter | Pearson $r$ | Physical Interpretation |
| :--- | :---: | :--- |
| **Temperature (`°F`)** | **`+0.7350`** | Direct radiative surface heating during daylight hours |
| **IsDaylight (`0/1`)** | **`+0.6812`** | Astronomical horizon gate (Sunrise to Sunset) |
| **Wind Direction (`°`)** | **`-0.2303`** | Diurnal shift between upslope daytime winds (`~30°–70°`) and downslope night winds (`~175°`) |
| **Humidity (`%`)** | **`-0.2262`** | High humidity corresponds to cloud cover and radiative attenuation |
| **Barometric Pressure (`Hg`)** | **`+0.1190`** | High pressure (`>30.44 Hg`) suppresses clouds; low pressure brings storm cover |
| **Wind Speed (`mph`)** | **`+0.0736`** | Slight midday convective breeze increase (`7.10 mph` at `12:00` vs. `5.33 mph` at `18:00`) |
| **Day Length (`hours`)** | **`+0.0735`** | Longer autumn days (`Sep/Oct`) yield higher daily solar harvest than `Dec` |

### B.3 What patterns can be observed between radiation, temperature, and humidity?
Radiation, Temperature, and Humidity interact across three distinct meteorological regimes:
1. **Dry Clear-Sky Regime (`Temp > 56 °F`, `Humidity < 65%`):** Unobstructed insolation produces high, stable daytime radiation (`800–1,085 W/m²`) and high daily average energy.
2. **Cool Overcast Regime (`Temp < 50 °F`, `Humidity > 85%`):** Stratus cloud decks reflect incoming shortwave radiation, holding daytime irradiance below `200 W/m²`.
3. **Humid Broken-Cloud Regime (`Temp 58–63 °F`, `Humidity 85–98%`):** Scattered cumulus clouds cause rapid oscillations between deep cloud shadows (`300–500 W/m²`) and brief, extreme cloud-edge lensing bursts (`1,300–1,601.26 W/m²`).

### B.4 Unusual Observations & Outliers
1. **Super-Solar Constant Spikes (`>1,361 W/m²`):** `49` observations exceed `1,200 W/m²`, peaking at **`1,601.26 W/m²` on `2016-09-04 12:15:04`**. In the 5-minute logs on `2016-09-04`, radiation dropped to `415.06 W/m²` at `12:10:07`, surged to **`1,601.26 W/m²` at `12:15:04`** (`93%` humidity), and dropped to `585.80 W/m²` at `12:20:08`. This is **Cloud-Edge Enhancement (Mie Scattering)**—where direct sunlight combines with strong diffuse reflection from cumulus cloud edges.
2. **Extreme Wind Gusts:** Wind speed reaches **`40.50 mph`** (`>9.8` standard deviations above the `6.24 mph` mean) during winter low-pressure fronts.
3. **Humidity Sensor Saturation (`101–103%`):** A handful of fog/condensation readings slightly exceed `100%` relative humidity due to capacitive sensor saturation during heavy mist.

### B.5 Bonus — Non-Obvious Insights Justified by Data
1. **Afternoon Orographic Cloud Asymmetry & Thermal Hysteresis Loop:** At symmetric solar angles (`09:00` vs. `15:00`), afternoon radiation (**`370.72 W/m²`**) is **32.7% lower** than morning radiation (**`550.66 W/m²`**) because daytime heating drives upslope moisture convergence, raising mean humidity from **`69.1%` at `09:00` to `78.8%` at `15:00`**. Simultaneously, thermal inertia keeps afternoon temperatures warmer (`56.11 °F` at `15:00` vs. `55.49 °F` at `09:00`), tracing a clockwise hysteresis loop.
2. **The Peak-vs.-Average Day Paradox:** **Zero** of the Top 5 days by *maximum* radiation appear in the Top 5 days by *average* radiation. `2016-09-04` logged the highest single reading (`1,601.26 W/m²`) at `93.88%` mean humidity, but its daily average was only **`153.58 W/m²`** (rank #84/118). Conversely, `2016-09-19` achieved the highest daily average (**`335.50 W/m²`**, `43.09%` mean humidity) with a smooth peak of `1,085.75 W/m²`.

---

## 4. Part C — Solar Day Analysis Report

### C.1 Highest Recorded Solar Radiation Event
- **Date of Peak Radiation:** `2016-09-04`
- **Exact Timestamp:** `12:15:04` (`2016-09-04 12:15:04`)
- **Maximum Radiation:** **`1,601.26 W/m²`**
- **Corresponding Environmental Telemetry:**
  - **Temperature:** `61 °F`
  - **Relative Humidity:** `93 %`
  - **Wind Speed:** `9.0 mph` (Wind Direction: `3.56°`)
  - **Barometric Pressure:** `30.47 Hg`
- **Sunrise & Sunset Comparison:**
  - **Sunrise:** `06:08:00` · **Sunset:** `18:35:00` (Total Daylight: `12 hours 27 minutes`)
  - **Time Elapsed After Sunrise:** **`6 hours, 7 minutes, 4 seconds`** (`6.12 hours` / `367.07 minutes`)
  - **Time Remaining Before Sunset:** **`6 hours, 19 minutes, 56 seconds`** (`6.33 hours`) — occurring just `6 minutes 26 seconds` prior to exact astronomical solar noon (`12:21:30`).

### C.2 Top 5 Days with Highest Average Solar Radiation
| Rank | Date | Daily Average Radiation (`W/m²`) | Daily Max Radiation (`W/m²`) | Mean Temp (`°F`) | Mean Humidity (`%`) | Sunrise | Sunset | Observations |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `2016-09-19` | **`335.50`** | `1,085.75` | `57.24` | `43.09%` | `06:11:00` | `18:22:00` | `275` |
| **2** | `2016-09-02` | **`333.47`** | `1,062.01` | `56.38` | `72.50%` | `06:07:00` | `18:37:00` | `279` |
| **3** | `2016-09-22` | **`325.52`** | `1,053.17` | `57.42` | `62.98%` | `06:11:00` | `18:19:00` | `284` |
| **4** | `2016-09-25` | **`319.92`** | `1,075.29` | `54.24` | `60.92%` | `06:12:00` | `18:16:00` | `284` |
| **5** | `2016-09-28` | **`312.40`** | `1,029.30` | `54.54` | `72.47%` | `06:13:00` | `18:14:00` | `282` |

---

## 5. Bonus — Machine Learning Solar Radiation Prediction

### 5.1 Feature Selection & Engineering
To predict `Radiation` (`W/m²`), we combined environmental measurements with engineered temporal geometry features:
- `SolarElevationProxy`: $\sin\left(\pi \cdot \frac{\text{TimeDecimal} - \text{SunriseDecimal}}{\text{DayLengthHours}}\right)$ during daylight (`0` at night)
- `Temperature` (`°F`), `Humidity` (`%`), `Pressure` (`Hg`), `Speed` (`mph`), `WindDirection(Degrees)`
- `TimeDecimal` (`0.0–24.0`), `DayLengthHours`, and `DayOfYear` (`245–366`)

### 5.2 Model Evaluation (`80/20` Train-Test Split: `26,148` Train / `6,538` Test)
| Model Architecture | Test RMSE (`W/m²`) | Test MAE (`W/m²`) | Test $R^2$ Score | Key Takeaway |
| :--- | :---: | :---: | :---: | :--- |
| **1. Linear Regression (Raw Features)** | `198.80` | `151.10` | `0.6032` | Fails to capture non-linear diurnal bell curve & night floor |
| **2. Physics-Informed Basis Regression** | `120.89` | `57.50` | `0.8533` | Adding `SolarElevationProxy` & interactions cuts RMSE by `39%` |
| **3. Random Forest Regressor (Ensemble)** | `83.90` | `34.32` | `0.9293` | Captures non-linear cloud/humidity thresholds (`Train R² = 0.9484`) |
| **4. Gradient Boosted Decision Trees (GBDT)** | **`81.82`** | **`33.57`** | **`0.9328`** | **Best overall model** — explains **`93.28%`** of variance |

### 5.3 Feature Usefulness Interpretation
1. **`SolarElevationProxy` (`62.60%` RF / `41.01%` GBDT):** Primary geometric determinant of clear-sky solar insolation envelope.
2. **`Temperature` (`19.11%` RF / `9.98%` GBDT):** Strongest real-time thermodynamic proxy for surface solar heating.
3. **`Humidity` (`8.19%` RF / `13.09%` GBDT):** Crucial for discounting clear-sky potential during afternoon cloud cover and fog.
4. **`TimeDecimal` & `DayOfYear` (`6.51%` RF / `15.95%` GBDT):** Capture morning-vs.-afternoon cloud asymmetry and autumn-to-winter solar declination decay.
5. **`DayLengthHours`, `Pressure`, `WindDirection`, `Speed` (`3.59%` RF / `19.96%` GBDT):** Refine predictions during synoptic weather fronts and trade-wind inversions.
