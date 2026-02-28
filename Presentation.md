# SHIELD: Scalable Hydrological Intelligence for Early flood-risk and Lead-time Detection

## 🌍 Brief about the Idea
**SHIELD** is an advanced AI-powered flood prediction and early warning system designed to bridge the gap between global climate data and local disaster preparedness. By combining high-resolution satellite imagery, real-time weather forecasts, and physics-informed environmental features (soil/terrain), SHIELD provides actionable flood risk assessments up to **15 days in advance**. 

The core mission is to empower local authorities and communities with enough lead time to implement "Local Defense" strategies, saving lives and protecting infrastructure from the increasingly unpredictable heavy rainfall events driven by climate change.

---

## 🚀 Opportunities & USP

### 1. How is it different from existing ideas?
Existing flood warning systems often rely on either purely statistical weather forecasts (which lack local terrain context) or hydrological models (which are compute-heavy and hard to scale). SHIELD uses a **Hybrid Neural-Gradient Boosting architecture**:
- **LSTM (Temporal context)**: Captures long-term rainfall patterns and soil saturation trends.
- **XGBoost (Spatial context)**: Interprets static features like USDA soil texture classes, elevation (SRTM), and proximity to water bodies.
- **Tiered Forecaster**: Unlike static models, SHIELD uses a recursive rolling window that adapts as "forecast" data turns into "historical" data.

### 2. How will it solve the problem?
SHIELD solves the **False Positive Problem** common in early warnings. By introducing **Data Source Awareness** (using an `is_forecast` feature flag), the model learns that forecast data is less certain. This allows for:
- **72-hour High-Confidence Warnings**: High precision for immediate action.
- **7-15 Day Outlook**: Early awareness for long-term logistics.

### 3. Unique Selling Proposition (USP)
- **Physics-Informed ML**: Not just a black box; it calculates dynamic "Flood Thresholds" based on real-world soil infiltration rates and antecedent precipitation.
- **Multi-Model Ensemble**: Integrates GFS (US) and ICON (German) forecast models to provide a robust, conservative "Worst Case" assessment.
- **Automated Feedback Loop**: Built-in comparison against Google Earth Engine (GEE) ground truth allows the system to detect "Model Drift" and self-trigger retraining.

---

## 🛠️ Features Offered
- **15-Day Rolling Forecast**: Daily updated flood probabilities for any target region.
- **Three-Tier Alert System**:
    - 🔴 **High Confidence Warning (Day 1-3)**: Imminent risk, immediate evacuation/protection.
    - 🟡 **Watch Advisory (Day 4-7)**: High likelihood pattern, monitor closely.
    - 🔵 **Outlook (Day 8-15)**: Statistical risk, preparatory planning.
- **Localized Terrain Analysis**: Integration of Digital Elevation Models (DEM) to identify low-lying risk zones.
- **Soil Saturation Tracking**: Real-time Antecedent Precipitation Index (API) calculation to determine soil's water-holding capacity.
- **Automated Operational Pipeline**: A daily `cron` job that fetches weather, runs predictions, and exports alerts without human intervention.

---

## 💻 Technologies Used
- **Programming Language**: Python 3.9+
- **Machine Learning**: 
    - **TensorFlow/Keras**: For the LSTM time-series embedding.
    - **XGBoost**: For the final classification and probability calibration.
    - **Scikit-learn**: For data scaling and preprocessing.
- **Data & Geospatial**:
    - **Google Earth Engine (GEE) API**: For GPM (Rainfall) and SRTM (Elevation) data.
    - **Open-Meteo API**: For ensemble weather forecasts (GFS/ICON).
- **Logic & Storage**: Pandas, NumPy, Joblib (Model serialization).

---

## ⚡ Usage of AMD Products/Solutions
To achieve the scale required for national-level deployment, SHIELD leverages AMD’s high-performance hardware ecosystem:

### 1. Training Acceleration (AMD Instinct™ & ROCm™)
The training of the LSTM model on thousands of regional data points is computationally intensive. Using **AMD Instinct™ MI series GPUs** with the **ROCm™ Open Software Platform**, training times can be reduced from hours to minutes.
- **Benefit**: Faster iteration on "Phase 5" continuous retraining triggers.

### 2. Large-Scale Data Preprocessing (AMD EPYC™)
GEE data extraction requires heavy parallel processing of CSVs and JSON responses. **AMD EPYC™ Processors** with high core counts (up to 128 cores) allow for parallel execution of multi-region `gee_simple.py` queries.
- **Benefit**: Capability to process 50+ regions simultaneously for daily operational runs.

### 3. Edge Deployment (AMD Ryzen™ Threadripper™)
For local state-level disaster management centres, **AMD Ryzen™ Threadripper™** workstations provide the necessary multi-threading to host the daily prediction pipeline and GIS visualization tools locally without needing constant cloud dependency.

---

## 💰 Estimated Implementation Cost (Optional)

| Component | Description | Estimated Monthly Cost |
|---|---|---|
| **Cloud Compute** | AMD-powered GPU instances (e.g., Azure HB/HC series) for training | $150 - $400 |
| **Operational Server** | Small Linux VM for daily cron jobs | $20 - $50 |
| **Data APIs** | Open-Meteo / IMD (Premium tiers for high frequency) | $0 - $100 |
| **Storage** | Model versions and historical GEE data | $10 - $30 |
| **Total (Opex)** | | **~$180 - $580 / month** |

---

## 📈 Future Enhancements
- **Live River Gauge Integration**: Directly feeding IoT sensor data into the `river_flood_risk` feature.
- **Mobile Alert App**: Extending the CSV alerts to a PWA (Progressive Web App) for ground-level responders.
- **Visual Heatmaps**: Converting tabular risk into interactive 2D maps using Folium/Leaflet.
