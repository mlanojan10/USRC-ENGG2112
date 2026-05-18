# Sydney Night-Time Cooling Failure MVP

This project predicts night-time cooling failure risk for selected Sydney suburbs using 2020 ERA5 hourly weather data and static suburb-level urban/geographic features.

Current MVP suburbs:
- Bondi
- Parramatta
- Campbelltown

Target:
Cooling failure = 1 if the day is a hot day and the overnight cooling rate is in the bottom 25% of hot-day cooling rates.

Night-time cooling rate:
(night_start_temp - night_min_temp) / hours from 18:00 to overnight minimum

Main model:
- Logistic Regression baseline
- Random Forest classifier

Main script:
scripts/sydney_3_suburbs_model.py

Main processed table:
data_processed/sydney_3_suburbs_feature_table_night_cooling.csv