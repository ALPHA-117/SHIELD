# SHIELD: System Diagrams & Architecture

This document provides a visual overview of the SHIELD (Scalable Hydrological Intelligence for Early flood-risk and Lead-time Detection) system, including its operational flow, user interactions, technical architecture, and UI design.

---

## 🔄 Process Flow Diagram
The operational lifecycle of SHIELD from data ingestion to automated feedback.

```mermaid
graph TD
    A[Start: Daily Cron Job] --> B{Data Acquisition}
    B -->|Historical GPM & Static SRTM| C[Google Earth Engine]
    B -->|Ensemble GFS/ICON| D[Open-Meteo API]
    
    C --> E[Feature Engineering]
    D --> E
    
    E -->|18 Physics-Informed Features| F[Hybrid Model Engine]
    subgraph "Hybrid Model Engine"
    F --> G[LSTM Layer: Temporal Context]
    G --> H[LSTM Probability Output]
    H --> I[XGBoost Layer: Spatial Context]
    I --> J[Final Probability Calibration]
    end
    
    J --> K{Risk Assessment}
    K -->|Prob >= 0.7| L[High Confidence Warning]
    K -->|Prob 0.4-0.6| M[Watch Advisory]
    K -->|Prob 0.1-0.3| N[Outlook Only]
    
    L & M & N --> O[Export Operational_Alerts/ CSV]
    O --> P[Disaster Response Action]
    
    P --> Q[30-Day Delay: GEE Ground Truth Sync]
    Q --> R[Automated Feedback Loop]
    R -->|Precision/Recall Drop?| S{Model Retraining Trigger}
    S -->|Yes| T[Phase 5: Automated Retraining Protocol]
    T --> F
    S -->|No| A

    style L fill:#f96,stroke:#333,stroke-width:2px
    style M fill:#ff9,stroke:#333,stroke-width:2px
    style N fill:#9cf,stroke:#333,stroke-width:2px
```

---

## 🏗️ System Architecture Diagram
Technical stack and data flow optimized for AMD hardware, merging high-level logic with system components.

```mermaid
graph TB
    subgraph "External Cloud Layers"
        GEE[Google Earth Engine]
        OM[Open-Meteo API]
        USDA[USDA Soil Database]
    end

    subgraph "SHIELD Core (AMD EPYC™ + Instinct™)"
        subgraph "A. Data Preprocessing"
            B(Data Preprocessing)
            DP2[Physics-Informed Features]
        end

        subgraph "B. Hybrid Model (ROCm™)"
            D{Hybrid Model}
            E[LSTM Layer: Temporal]
            F[XGBoost Layer: Spatial]
        end
    end

    subgraph "C. Monitoring & Feedback"
        G[Flood Probability Engine]
        H[Three-Tier Alert System]
        I[Operational CSV Reports]
        J[Feedback Loop vs GEE GT]
    end

    %% Flow Connections with detailed labels
    GEE -->|GPM Rainfall & SRTM Elevation| B
    OM -->|GFS & ICON Forecasts| B
    USDA --> DP2
    
    B --> D
    DP2 --> D
    
    D -->|Time-Series Patterns| E
    D -->|Spatial Features| F
    
    E --> G
    F --> G
    
    G --> H
    H -->|🔴 🟡 🔵| I
    I --> J
    J -->|Retraining Trigger| D

    style D fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#fff,stroke:#333
```

---

## 👥 Use Case Diagram
Interaction between different stakeholders and the SHIELD system.

```mermaid
graph LR
    subgraph "SHIELD System"
        UC1(Monitor Regional Flood Risk)
        UC2(Receive Early Warning Alerts)
        UC3(Analyze Historical Metrics)
        UC4(Handle Model Drift & Retraining)
        UC5(Configure Regional Parameters)
    end

    Actor1((Local Defense / First Responders))
    Actor2((Disaster Management Authority))
    Actor3((System Admin / Data Scientist))

    Actor1 --> UC2
    
    Actor2 --> UC1
    Actor2 --> UC2
    Actor2 --> UC3
    
    Actor3 --> UC4
    Actor3 --> UC5
    Actor3 --> UC3
```

---

## 🎨 UI Wireframe Mockup
Proposed dashboard layout for the "High Confidence Warning" view.

```mermaid
graph TD
    subgraph Dashboard ["SHIELD Dashboard v1.0"]
        Header["SHIELD: Flood Warning System"]
        
        subgraph RegionPanel ["Region: Barpeta, Assam"]
            Status["HIGH CONFIDENCE WARNING (72h)"]
            Prob["Risk Probability: 86%"]
            
            subgraph ChartArea ["Main Chart"]
                Chart["--- Probability 15-Day Forecast Graph ---"]
            end
            
            subgraph ContextArea ["Environmental Context"]
                Soil["Soil: Clay Loam"]
                Sat["Saturation: 92%"]
                Rain["Forecast Rain: 42mm"]
            end
        end
        
        Footer["Export Report | Model Feedback | Settings"]
    end

    Header --- Status
    Status --- Prob
    Prob --- ChartArea
    ChartArea --- ContextArea
    ContextArea --- Footer

    style Status fill:#f66,stroke:#333,stroke-width:4px
    style Dashboard fill:#eee,stroke:#333
    style RegionPanel fill:#fff,stroke:#ccc
```
