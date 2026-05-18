# Sydney Night-Time Cooling Failure MVP

This project predicts night-time cooling failure risk across selected Sydney suburbs using 2020 ERA5 hourly weather data and static suburb-level urban/geographic features.

The aim is to identify suburb-days where overnight cooling is poor, because high night-time temperatures can increase health risk by reducing recovery from daytime heat exposure.

## Current study area

The final MVP uses 12 Sydney suburbs across a coastal-to-inland gradient:

- Coastal / water-adjacent: Bondi, Manly, Coogee, Cronulla
- Middle inland: Parramatta, Ryde, Bankstown, Fairfield
- Far inland / western-southwestern: Penrith, Blacktown, Liverpool, Campbelltown

Each row in the final model table represents one suburb on one day.

## Target variable

The target variable is cooling_failure.

Cooling failure is defined as:

    cooling_failure = 1 if:
        overnight cooling rate is in the slowest-cooling 25% of all observed suburb-nights

    cooling_failure = 0 otherwise

The overnight cooling rate is calculated as:

    night_cooling_rate =
    (night_start_temp - night_min_temp) / hours from 18:00 to overnight minimum

This means the model is not simply predicting whether a day was hot. Instead, it predicts whether a suburb cools down poorly overnight.

In the current model run:

    Cooling failure threshold = 0.2023 °C/hour

So nights cooling at approximately 0.20°C/hour or slower are labelled as cooling failures.

## Hot day variable

A hot day is still calculated using:

    hot_day = 1 if tmax >= 30°C

However, hot_day is now used only as a context variable. It is not used to define the target variable.

This change was made because the main purpose of the model is to predict night-time cooling failure, not simply daytime heat.

## Data sources

### Weather data

Weather variables are taken from ERA5 hourly time-series data for 2020.

Processed daily weather features include:

- daily maximum temperature, tmax
- daily minimum temperature, tmin
- daily temperature range, daily_temp_range
- mean dewpoint, dewpoint_mean
- mean humidity, humidity_mean
- mean wind speed, wind_speed_mean
- mean air pressure, air_pressure_mean
- total precipitation, precipitation_total

### Static suburb features

Static features are merged from:

    data_processed/suburb_static_features.csv

Modelled static features include:

- distance to coast
- distance to nearest major water body
- coastal status
- tree canopy percentage
- population density
- built-up area percentage

Raw size-based columns such as population count, area, tree canopy area, and built-up area are saved for reporting but are not used as model predictors.

## Features used for modelling

The final model uses:

    tmax
    tmin
    daily_temp_range
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
    built_up_area_percent
    month
    is_summer

The model does not use the following as predictors because they would cause target leakage:

    night_cooling_rate
    night_start_temp
    night_min_temp
    cooling_failure
    cooling_risk_category

## Machine learning workflow

The project uses supervised binary classification.

The model predicts:

    0 = normal / better overnight cooling
    1 = cooling failure / slow overnight cooling

The workflow is:

1. Load hourly ERA5 weather data.
2. Convert hourly weather data into daily suburb-level features.
3. Calculate night-time cooling rate from 18:00 to the overnight minimum.
4. Label the slowest-cooling 25% of nights as cooling failures.
5. Merge static suburb-level urban/geographic features.
6. Train a Logistic Regression baseline model.
7. Train a Random Forest main model.
8. Evaluate model performance using a train/test split and 5-fold cross-validation.
9. Save model outputs, feature importance, predictions, and summary tables.

## Models used

### Logistic Regression baseline

Logistic Regression is used as a simple baseline model.

It estimates the probability of cooling failure using a weighted combination of the input features. In simple terms, it acts like a basic scoring system that tries to separate normal-cooling nights from slow-cooling nights.

The current Logistic Regression test results are:

    Accuracy: 0.83
    Cooling failure precision: 0.61
    Cooling failure recall: 0.88
    Cooling failure F1-score: 0.72

### Random Forest main model

Random Forest is used as the main model because it can handle nonlinear relationships and interactions between weather, geographic, and urban-form variables.

It works by building many decision trees and combining their predictions. In simple terms, it is like asking many small decision-makers to vote on whether a suburb-day is likely to experience poor overnight cooling.

The current Random Forest test results are:

    Accuracy: 0.89
    Cooling failure precision: 0.72
    Cooling failure recall: 0.89
    Cooling failure F1-score: 0.79

The Random Forest model performs better than Logistic Regression overall, especially in balancing detection of cooling-failure nights with fewer false alarms.

## Cross-validation results

A 5-fold stratified cross-validation step is used on the training set to check whether the Random Forest performance is stable.

Current mean cross-validation results:

    Mean accuracy: 0.878
    Mean precision: 0.709
    Mean recall: 0.877
    Mean F1-score: 0.783
    Mean ROC-AUC: 0.944

This suggests the model performance is reasonably consistent across different training splits, rather than being caused by one lucky train/test split.

## Confusion matrix interpretation

The current Random Forest confusion matrix can be interpreted as:

    Correctly predicted normal-cooling nights: 875
    False alarms: 114
    Missed cooling-failure nights: 37
    Correctly detected cooling-failure nights: 292

This means the model detected most slow-cooling nights.

For a heat-risk warning application, recall is especially important because missing a risky slow-cooling night is more harmful than issuing some false alarms.

## Feature importance

The most important Random Forest features in the current run are:

    daily_temp_range
    distance_to_water_km
    precipitation_total
    distance_to_coast_km
    is_coastal
    wind_speed_mean
    humidity_mean

This suggests that poor overnight cooling is influenced by both short-term weather conditions and geographic location.

Feature importance does not prove direct causation, but it helps explain which variables were most useful for prediction.

## Current outputs

The script creates:

    data_processed/sydney_12_suburbs_feature_table_night_cooling.csv

    outputs/tables/feature_importance_sydney_12_suburbs.csv
    outputs/tables/model_predictions_sydney_12_suburbs.csv
    outputs/tables/suburb_summary_sydney_12_suburbs.csv
    outputs/tables/risk_category_counts_sydney_12_suburbs.csv
    outputs/tables/cross_validation_scores_sydney_12_suburbs.csv

    outputs/figures/confusion_matrix_sydney_12_suburbs.png
    outputs/figures/feature_importance_sydney_12_suburbs.png

## Current interpretation

The current MVP predicts night-time cooling failure across 12 Sydney suburbs using daily weather and static suburb-level features.

The final target is based directly on overnight cooling rate, which better matches the project aim than the earlier hot-day-based target.

The Random Forest model performs better than the Logistic Regression baseline and achieves strong recall for cooling-failure nights. This means the model is reasonably effective at identifying suburb-days where overnight cooling is poor.

The main predictors suggest that cooling failure is associated with daily temperature range, water/coastal proximity, precipitation, wind, humidity, and urban/geographic factors.

## Key limitations

- The model currently uses only one year of ERA5 data: 2020.
- ERA5 is gridded weather data, so it may not fully capture suburb-level microclimates.
- Static suburb features do not vary over time.
- The target is based on relative cooling rate, not direct health outcome data.
- Slow cooling does not always mean dangerously high night-time temperature; it is a proxy for overnight heat retention.
- More years of data would improve generalisation.
- Future work could include observed weather station data, additional years, land surface temperature, NDVI, roof colour, or health/emergency response data.

## Final modelling note

The current model should be described as a night-time cooling failure classifier, not a general heatwave model.

It predicts whether a suburb-day is likely to experience poor overnight cooling based on weather, geographic, and urban-form conditions.
