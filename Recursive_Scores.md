# SHIELD Recursive Rolling Evaluation (Cron Job Simulation)
    
Based on generating rolling horizon forecasts against 50 historical regions where the model incrementally updates its context daily.
Total evaluation prediction instances: 6000

## Advance Warning Detection Metrics
*How early did the model accurately classify the danger before it happened?*

- **1-Day Lead Time** (threshold=0.35): Accuracy: 0.940 | Precision: 0.381 | Recall: 0.457 | F1: 0.416 (Total Floods: 35)
- **3-Day Lead Time** (threshold=0.40): Accuracy: 0.940 | Precision: 0.318 | Recall: 0.226 | F1: 0.264 (Total Floods: 31)
- **5-Day Lead Time** (threshold=0.45): Accuracy: 0.951 | Precision: 0.500 | Recall: 0.296 | F1: 0.372 (Total Floods: 27)
- **7-Day Lead Time** (threshold=0.45): Accuracy: 0.949 | Precision: 0.500 | Recall: 0.391 | F1: 0.439 (Total Floods: 23)
- **10-Day Lead Time** (threshold=0.50): Accuracy: 0.953 | Precision: 0.429 | Recall: 0.231 | F1: 0.300 (Total Floods: 13)
- **15-Day Lead Time** (threshold=0.50): Accuracy: 1.000 | Precision: 1.000 | Recall: 1.000 | F1: 1.000 (Total Floods: 3)