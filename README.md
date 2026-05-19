# Sydney Night-Time Heat-Health Risk MVP

This project predicts night-time heat-health risk across selected Sydney suburbs using 2020 ERA5 hourly weather data and static suburb-level urban/geographic features.

The final project focus is health-related night-time heat risk, not just daytime heat or broad urban heat island classification. The model aims to identify suburb-days where residents may have reduced overnight recovery from heat because the suburb both cooled slowly and stayed warm overnight.

## Current study area

The MVP uses 12 Sydney suburbs across a coastal-to-inland gradient:

- Coastal / water-adjacent: Bondi, Manly, Coogee, Cronulla
- Middle inland: Parramatta, Ryde, Bankstown, Fairfield
- Far inland / western-southwestern: Penrith, Blacktown, Liverpool, Campbelltown

Each row in the final model table represents one suburb on one day.

## Final project framing

The project now uses two related modelling layers:

1. Supporting cooling-rate analysis
   - Measures whether a suburb cooled slowly overnight.
   - This is useful for understanding cooling behaviour.

2. Final health-risk model
   - Predicts whether a suburb cooled slowly and also stayed warm overnight.
   - This is the main model for the health-risk interpretation.

The final headline model is the night-time heat-health risk model.

## Target variables

### Supporting target: cooling_failure

Cooling failure is defined as:

    cooling_failure = 1 if:
        overnight cooling rate is in the slowest-cooling 25% of all observed suburb-nights

    cooling_failure = 0 otherwise

The overnight cooling rate is calculated as:

    night_cooling_rate =
    (night_start_temp - night_min_temp) / hours from 18:00 to overnight minimum

In the current model run:

    Cooling failure threshold = 0.2023 °C/hour

So nights cooling at approximately 0.20°C/hour or slower are labelled as cooling failures.

### Final target: night_heat_health_risk

The final health-risk target is:

    night_heat_health_risk = 1 if:
        cooling_failure = 1
        AND
        night_min_temp >= 20°C

    night_heat_health_risk = 0 otherwise

This means a suburb-night is only labelled as heat-health risk if it both cooled slowly and stayed warm overnight.

This was added because slow cooling alone can overstate risk in coastal suburbs. A coastal suburb may cool slowly because ocean temperatures moderate day-night variation, but that does not always mean the night is dangerously hot. The health-risk target is therefore more relevant to the original project aim.

## Hot day variable

A hot day is still calculated using:

    hot_day = 1 if tmax >= 30°C

However, hot_day is used only as a context variable. It is not used to define the final health-risk target.

## Data sources

### Weather data

Weather variables are taken from ERA5 hourly time-series data for 2020.

Processed daily weather features include:

- daily maximum temperature, tmax
- mean dewpoint, dewpoint_mean
- mean humidity, humidity_mean
- mean wind speed, wind_speed_mean
- mean air pressure, air_pressure_mean
- total precipitation, precipitation_total

The final model excludes tmin and daily_temp_range from the predictor set because they are closely related to the target calculation.

### Static suburb features

Static features are merged from:

    data_processed/suburb_static_features.csv

Modelled static features include:

- distance to coast
- distance to nearest major water body
- coastal status
- tree canopy percentage
- population density

built_up_area_percent is saved for reporting, but excluded from the final predictor set because some SA2 proxy values were inflated near 100%.

## Features used for modelling

The final defensible predictor set uses:

    tmax
    dewpoint_mean
    humidity_mean
    wind_speed_mean
    air_pressure_mean
    precipitation_total
    distance_to_coast_km
    distance_to_water_km
    is_coastal
    tree_canopy_percent
    population_density_people_per_km2
    month
    is_summer

The model does not use the following as predictors because they could cause leakage or data-quality issues:

    tmin
    daily_temp_range
    built_up_area_percent
    night_cooling_rate
    night_start_temp
    night_min_temp
    cooling_failure
    cooling_risk_category
    night_heat_health_risk

## Machine learning workflow

The project uses supervised binary classification.

The final model predicts:

    0 = no night-time heat-health risk
    1 = night-time heat-health risk

The workflow is:

1. Load hourly ERA5 weather data.
2. Convert hourly weather data into daily suburb-level features.
3. Calculate night-time cooling rate from 18:00 to the overnight minimum.
4. Create the supporting cooling_failure target using the slowest-cooling 25% threshold.
5. Create the final night_heat_health_risk target using slow cooling plus night_min_temp >= 20°C.
6. Merge static suburb-level urban/geographic features.
7. Remove leakage-adjacent predictors from the model input.
8. Train a Logistic Regression baseline model.
9. Train a Random Forest main model.
10. Evaluate model performance using a 70/30 train/test split and 5-fold cross-validation.
11. Save model outputs, feature importance, predictions, and summary tables.

## Train, validation, and test setup

The model uses:

    Training set: 70%
    Testing set: 30%

Validation is performed using 5-fold cross-validation on the training set.

Current split:

    Total model rows: 4392
    Training rows: 3074
    Testing rows: 1318

For the final health-risk target:

    Training no-risk cases: 2844
    Training risk cases: 230

    Testing no-risk cases: 1220
    Testing risk cases: 98

## Models used

### Logistic Regression baseline

Logistic Regression is used as a simple baseline model.

It estimates the probability of the target using a weighted combination of the input features. In simple terms, it acts like a basic scoring system that tries to separate low-risk nights from risk nights.

Current Logistic Regression test results for night-time heat-health risk:

    Accuracy: 0.894
    Night heat-health precision: 0.406
    Night heat-health recall: 0.929
    Night heat-health F1-score: 0.565

### Random Forest main model

Random Forest is used as the main model because it can handle nonlinear relationships and interactions between weather, geographic, and urban-form variables.

It works by building many decision trees and combining their predictions. In simple terms, it is like asking many small decision-makers to vote on whether a suburb-day is likely to experience night-time heat-health risk.

Current Random Forest test results for night-time heat-health risk:

    Accuracy: 0.956
    Night heat-health precision: 0.630
    Night heat-health recall: 0.990
    Night heat-health F1-score: 0.770
    ROC-AUC: 0.990

The Random Forest model performs better than Logistic Regression overall and catches almost all night heat-health risk cases in the test set.

## Cross-validation results

A 5-fold stratified cross-validation step is used on the training set to check whether the Random Forest performance is stable.

Current mean cross-validation results for the night-time heat-health risk model:

    Mean accuracy: 0.941
    Mean precision: 0.565
    Mean recall: 0.952
    Mean F1-score: 0.709
    Mean ROC-AUC: 0.985

This suggests the model performs consistently across different training splits.

## Confusion matrix interpretation

The current Random Forest confusion matrix for night-time heat-health risk can be interpreted as:

    Correctly predicted no night heat-health risk: 1163
    False alarms: 57
    Missed night heat-health risk cases: 1
    Correctly detected night heat-health risk cases: 97

This means the model detected 97 out of 98 heat-health risk cases in the test set.

For a health-risk warning application, recall is especially important because missing a risky night is more harmful than issuing some false alarms.

## Feature importance

The most important Random Forest features in the final night-time heat-health risk model are:

    dewpoint_mean
    tmax
    month
    is_summer
    humidity_mean
    distance_to_coast_km
    distance_to_water_km
    wind_speed_mean
    precipitation_total
    is_coastal
    air_pressure_mean
    tree_canopy_percent
    population_density_people_per_km2

This suggests that warm, slow-cooling nights are most strongly associated with moisture, daytime heat, seasonality, humidity, and geographic/coastal influences.

Feature importance does not prove direct causation. It only indicates which variables were most useful to the Random Forest model for prediction.

## Suburb-level interpretation

The final health-risk model separates slow cooling from warm-night health risk.

In the current output, the number of night heat-health risk days is:

    Bondi: 69
    Coogee: 69
    Cronulla: 69
    Manly: 46
    Bankstown: 14
    Liverpool: 14
    Blacktown: 10
    Fairfield: 10
    Parramatta: 10
    Ryde: 10
    Penrith: 4
    Campbelltown: 3

Important caution: these suburb-level values should not be interpreted as exact real-world suburb rankings. Some suburbs share identical ERA5 weather time series because ERA5 is gridded and coarser than suburb boundaries.

The suburb table is best interpreted as a model output summary, not a validated public-health ranking.

## Data quality warnings and limitations

### Leakage-adjacent predictors removed

tmin and daily_temp_range were removed from the final predictor set.

This was done because:

- tmin is closely related to the overnight minimum temperature used in the target calculation.
- daily_temp_range = tmax - tmin, so it indirectly encodes tmin.
- Keeping these variables allowed the model to learn target-like temperature shortcuts.

After these changes, the cooling-rate model became slightly less accurate, but the result became more defensible.

### Built-up area removed from predictors

built_up_area_percent was removed from the final predictor set because some suburbs had values close to or equal to 100%, likely due to SA2 proxy limitations.

The variable is still saved for reporting, but it is treated as a broad urban development proxy rather than literal building footprint coverage.

### ERA5 grid-cell limitation

Some suburbs share identical ERA5 weather time series because ERA5 is gridded and relatively coarse compared with suburb boundaries.

Detected shared ERA5 groups:

    Bondi, Coogee, Cronulla
    Bankstown, Liverpool
    Blacktown, Fairfield, Parramatta, Ryde

This limits suburb-level microclimate interpretation.

### Bankstown population correction

The raw population-density file originally contained an incorrect Bankstown proxy match to Banksmeadow.

The corrected final static feature table uses:

    Bankstown - North
    population density around 6471.6 people/km²

## Current outputs

The cooling-rate model creates:

    data_processed/sydney_12_suburbs_feature_table_night_cooling.csv

    outputs/tables/feature_importance_sydney_12_suburbs.csv
    outputs/tables/model_predictions_sydney_12_suburbs.csv
    outputs/tables/suburb_summary_sydney_12_suburbs.csv
    outputs/tables/risk_category_counts_sydney_12_suburbs.csv
    outputs/tables/cross_validation_scores_sydney_12_suburbs.csv
    outputs/tables/final_model_metrics_summary.csv
    outputs/tables/duplicate_era5_weather_groups.csv

    outputs/figures/confusion_matrix_sydney_12_suburbs.png
    outputs/figures/feature_importance_sydney_12_suburbs.png

The final heat-health model creates:

    outputs/tables/night_heat_health_risk_metrics_summary.csv
    outputs/tables/night_heat_health_risk_predictions.csv
    outputs/tables/night_heat_health_risk_suburb_summary.csv
    outputs/tables/night_heat_health_risk_feature_importance.csv

    outputs/figures/night_heat_health_risk_confusion_matrix.png
    outputs/figures/night_heat_health_risk_feature_importance.png

## Final interpretation

The current MVP predicts night-time heat-health risk across 12 Sydney suburbs using daily weather and static suburb-level features.

The supporting cooling-rate model identifies nights where suburbs cooled slowly. The final health-risk model refines this by requiring both slow cooling and an overnight minimum temperature of at least 20°C.

This is more aligned with the original health motivation because slow cooling alone does not necessarily mean dangerous heat exposure.

The final Random Forest model achieved 0.956 accuracy, 0.630 precision, 0.990 recall and 0.770 F1-score for night-time heat-health risk on the held-out test set.

The model should be described as a night-time heat-health risk classifier, not a general heatwave model or a fully validated public-health warning system.
