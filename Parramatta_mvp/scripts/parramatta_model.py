from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# =========================
# 0. PROJECT PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data_processed"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
TABLES_DIR = BASE_DIR / "outputs" / "tables"

PROCESSED_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# This file should be created by your Meteostat extraction script.
# Expected columns:
# datetime, suburb, station_id, station_name, temperature,
# humidity, wind_speed, air_pressure, precipitation
HOURLY_FILE = PROCESSED_DIR / "parramatta_hourly_2020.csv"

FEATURE_TABLE_FILE = PROCESSED_DIR / "parramatta_feature_table_night_cooling.csv"

FEATURE_IMPORTANCE_FILE = TABLES_DIR / "feature_importance_night_cooling.csv"
MODEL_PREDICTIONS_FILE = TABLES_DIR / "model_predictions_night_cooling.csv"

CONFUSION_MATRIX_FIG = FIGURES_DIR / "confusion_matrix_night_cooling.png"
FEATURE_IMPORTANCE_FIG = FIGURES_DIR / "feature_importance_night_cooling.png"


# =========================
# 1. CHECK INPUT FILE
# =========================

if not HOURLY_FILE.exists():
    raise FileNotFoundError(f"Missing required hourly file: {HOURLY_FILE}")


# =========================
# 2. LOAD HOURLY METEOSTAT DATA
# =========================

df = pd.read_csv(HOURLY_FILE)

print("Original hourly columns:")
print(df.columns)

print("\nFirst 5 hourly rows:")
print(df.head())


# =========================
# 3. BASIC CLEANING
# =========================

required_cols = ["datetime", "temperature"]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column in hourly file: {col}")

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")

# Keep everything consistent with Meteostat.
# These columns should all come from the same hourly Meteostat file.
meteostat_weather_cols = [
    "humidity",
    "wind_speed",
    "air_pressure",
    "precipitation",
]

for col in meteostat_weather_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        print(f"Warning: optional Meteostat column missing: {col}")

df = df.dropna(subset=["datetime", "temperature"]).copy()

if "suburb" not in df.columns:
    df["suburb"] = "Parramatta"

if "station_id" not in df.columns:
    df["station_id"] = "Unknown"

if "station_name" not in df.columns:
    df["station_name"] = "Unknown"

# IMPORTANT:
# The Meteostat hourly timestamps may be in UTC depending on the source.
# This converts them to Sydney local time for night-time analysis.
#
# If your extracted file is already definitely Sydney local time, you can comment
# this block out. But for night-time cooling, local time is usually safer.
if df["datetime"].dt.tz is None:
    df["datetime"] = (
        df["datetime"]
        .dt.tz_localize("UTC")
        .dt.tz_convert("Australia/Sydney")
        .dt.tz_localize(None)
    )
else:
    df["datetime"] = (
        df["datetime"]
        .dt.tz_convert("Australia/Sydney")
        .dt.tz_localize(None)
    )

df["date"] = df["datetime"].dt.floor("D")
df["hour"] = df["datetime"].dt.hour
df["month"] = df["datetime"].dt.month
df["is_summer"] = df["month"].isin([12, 1, 2]).astype(int)

print("\nDatetime range after Sydney-time conversion:")
print(df["datetime"].min(), "to", df["datetime"].max())

print("\nHour counts:")
print(df["hour"].value_counts().sort_index())


# =========================
# 4. DEFINE NIGHT-TIME PERIOD
# =========================

# Night period = 6 pm to 6 am.
#
# Example:
# 2020-01-01 18:00 to 2020-01-02 05:00
# belongs to night_date = 2020-01-01.
#
# This avoids accidentally treating midnight as the start of the night.

df["is_night"] = (df["hour"] >= 18) | (df["hour"] < 6)

df["night_date"] = df["date"]

early_morning_mask = df["hour"] < 6
df.loc[early_morning_mask, "night_date"] = (
    df.loc[early_morning_mask, "date"] - pd.Timedelta(days=1)
)

night_df = df[df["is_night"]].copy()

if night_df.empty:
    raise ValueError("No night-time rows found. Check datetime/hour parsing.")

print("\nNight-time row preview:")
print(
    night_df[
        ["datetime", "date", "hour", "night_date", "temperature"]
    ].head(15)
)


# =========================
# 5. DAILY FEATURES FROM METEOSTAT
# =========================

daily = df.groupby("date").agg(
    tmax=("temperature", "max"),
    tmin=("temperature", "min"),
    daily_temp_range=("temperature", lambda x: x.max() - x.min()),
).reset_index()

# Daily mean humidity from Meteostat
if "humidity" in df.columns:
    humidity_daily = df.groupby("date")["humidity"].mean().reset_index()
    humidity_daily = humidity_daily.rename(columns={"humidity": "humidity_mean"})
    daily = daily.merge(humidity_daily, on="date", how="left")

# Daily mean wind speed from Meteostat
if "wind_speed" in df.columns:
    wind_daily = df.groupby("date")["wind_speed"].mean().reset_index()
    wind_daily = wind_daily.rename(columns={"wind_speed": "wind_speed_mean"})
    daily = daily.merge(wind_daily, on="date", how="left")

# Daily mean air pressure from Meteostat
if "air_pressure" in df.columns:
    pressure_daily = df.groupby("date")["air_pressure"].mean().reset_index()
    pressure_daily = pressure_daily.rename(columns={"air_pressure": "air_pressure_mean"})
    daily = daily.merge(pressure_daily, on="date", how="left")

# Daily total precipitation from Meteostat
if "precipitation" in df.columns:
    precipitation_daily = df.groupby("date")["precipitation"].sum(min_count=1).reset_index()
    precipitation_daily = precipitation_daily.rename(
        columns={"precipitation": "precipitation_total"}
    )
    daily = daily.merge(precipitation_daily, on="date", how="left")
else:
    daily["precipitation_total"] = 0

daily["precipitation_total"] = daily["precipitation_total"].fillna(0)

print("\nDaily Meteostat feature preview:")
print(daily.head())


# =========================
# 6. NIGHT-TIME COOLING RATE
# =========================

def calculate_night_cooling(group):
    group = group.sort_values("datetime").copy()

    # Start-of-night should be the first available reading at or after 18:00.
    # This avoids choosing midnight or early morning as the "start".
    evening_rows = group[group["hour"] >= 18]

    if evening_rows.empty or len(group) < 4:
        return pd.Series(
            {
                "night_start_time": pd.NaT,
                "night_min_time": pd.NaT,
                "night_start_temp": pd.NA,
                "night_min_temp": pd.NA,
                "night_hours_to_min": pd.NA,
                "night_total_hours": pd.NA,
                "night_cooling_rate": pd.NA,
            }
        )

    night_start_time = evening_rows["datetime"].iloc[0]
    night_start_temp = evening_rows["temperature"].iloc[0]

    # Only use temperatures after the actual start-of-night.
    after_start = group[group["datetime"] >= night_start_time].copy()

    if after_start.empty:
        return pd.Series(
            {
                "night_start_time": pd.NaT,
                "night_min_time": pd.NaT,
                "night_start_temp": pd.NA,
                "night_min_temp": pd.NA,
                "night_hours_to_min": pd.NA,
                "night_total_hours": pd.NA,
                "night_cooling_rate": pd.NA,
            }
        )

    night_min_idx = after_start["temperature"].idxmin()
    night_min_time = after_start.loc[night_min_idx, "datetime"]
    night_min_temp = after_start.loc[night_min_idx, "temperature"]

    night_end_time = after_start["datetime"].iloc[-1]

    night_hours_to_min = (
        night_min_time - night_start_time
    ).total_seconds() / 3600

    night_total_hours = (
        night_end_time - night_start_time
    ).total_seconds() / 3600

    # If the minimum occurs immediately at the start, cooling rate is 0.
    # This can happen during warming nights or incomplete records.
    if night_hours_to_min <= 0:
        night_cooling_rate = 0.0
    else:
        night_cooling_rate = (
            night_start_temp - night_min_temp
        ) / night_hours_to_min

    return pd.Series(
        {
            "night_start_time": night_start_time,
            "night_min_time": night_min_time,
            "night_start_temp": night_start_temp,
            "night_min_temp": night_min_temp,
            "night_hours_to_min": night_hours_to_min,
            "night_total_hours": night_total_hours,
            "night_cooling_rate": night_cooling_rate,
        }
    )


night_features = (
    night_df
    .groupby("night_date", group_keys=False)
    .apply(calculate_night_cooling, include_groups=False)
    .reset_index()
)

night_features = night_features.rename(columns={"night_date": "date"})
night_features["date"] = pd.to_datetime(night_features["date"], errors="coerce")

for col in [
    "night_start_temp",
    "night_min_temp",
    "night_hours_to_min",
    "night_total_hours",
    "night_cooling_rate",
]:
    night_features[col] = pd.to_numeric(night_features[col], errors="coerce")

print("\nNight cooling feature preview:")
print(
    night_features[
        [
            "date",
            "night_start_time",
            "night_min_time",
            "night_start_temp",
            "night_min_temp",
            "night_hours_to_min",
            "night_total_hours",
            "night_cooling_rate",
        ]
    ].head(10)
)


# =========================
# 7. MERGE DAILY + NIGHT FEATURES
# =========================

df_model = daily.merge(
    night_features,
    on="date",
    how="inner",
)

df_model["suburb"] = "Parramatta"
df_model["month"] = df_model["date"].dt.month
df_model["is_summer"] = df_model["month"].isin([12, 1, 2]).astype(int)

df_model = df_model.dropna(
    subset=[
        "tmax",
        "tmin",
        "night_start_temp",
        "night_min_temp",
        "night_hours_to_min",
        "night_total_hours",
        "night_cooling_rate",
    ]
).copy()

print("\nMerged model table preview:")
print(
    df_model[
        [
            "date",
            "tmax",
            "tmin",
            "night_start_temp",
            "night_min_temp",
            "night_hours_to_min",
            "night_cooling_rate",
        ]
    ].head(10)
)


# =========================
# 8. CREATE HOT DAY FLAG
# =========================

# Hot day = top 30% hottest days by maximum temperature.
hot_threshold = df_model["tmax"].quantile(0.70)

df_model["hot_day"] = (df_model["tmax"] >= hot_threshold).astype(int)

print("\nHot day threshold:")
print(hot_threshold)

print("\nHot day counts:")
print(df_model["hot_day"].value_counts())


# =========================
# 9. CREATE TARGET VARIABLE
# =========================

hot_days = df_model[df_model["hot_day"] == 1].copy()

if len(hot_days) < 10:
    raise ValueError(
        "Not enough hot-day samples. Use more weather data or lower the hot-day threshold."
    )

# Cooling failure = hot day where overnight cooling rate is in the bottom 25%
# of hot-day cooling rates.
#
# Lower cooling rate = slower cooling = worse overnight cooling.
cooling_failure_threshold = hot_days["night_cooling_rate"].quantile(0.25)

df_model["cooling_failure"] = (
    (df_model["hot_day"] == 1)
    & (df_model["night_cooling_rate"] <= cooling_failure_threshold)
).astype(int)

print("\nCooling failure threshold:")
print(cooling_failure_threshold)

print("\nTarget class counts:")
print(df_model["cooling_failure"].value_counts())


# =========================
# 10. RISK CATEGORIES
# =========================

# Based on the project target definition:
# Top 50% fastest cooling       = Low risk
# 25th to 50th percentile       = Moderate risk
# Bottom 10th to 25th percentile = High risk
# Bottom 10% slowest cooling    = Extreme risk

p10 = hot_days["night_cooling_rate"].quantile(0.10)
p25 = hot_days["night_cooling_rate"].quantile(0.25)
p50 = hot_days["night_cooling_rate"].quantile(0.50)

def assign_risk_category(row):
    if row["hot_day"] != 1:
        return "Not hot day"

    rate = row["night_cooling_rate"]

    if rate <= p10:
        return "Extreme risk"
    elif rate <= p25:
        return "High risk"
    elif rate <= p50:
        return "Moderate risk"
    else:
        return "Low risk"


df_model["cooling_risk_category"] = df_model.apply(assign_risk_category, axis=1)

print("\nCooling risk category counts:")
print(df_model["cooling_risk_category"].value_counts())


# =========================
# 11. PREPARE FEATURE COLUMNS
# =========================

# IMPORTANT:
# Do NOT include night_cooling_rate in the ML features because it directly defines the label.
# Do NOT include hot_day because cooling_failure is only defined on hot days.
# Do NOT include night_min_temp if you want to avoid strong leakage from the target calculation.
#
# We keep the feature set based on day conditions and Meteostat weather variables.

possible_features = [
    "tmax",
    "tmin",
    "daily_temp_range",
    "humidity_mean",
    "wind_speed_mean",
    "air_pressure_mean",
    "precipitation_total",
    "month",
    "is_summer",
]

features = [col for col in possible_features if col in df_model.columns]

for col in features:
    df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

    if df_model[col].isna().all():
        print(f"Warning: dropping feature because it is entirely missing: {col}")
        features.remove(col)
    else:
        df_model[col] = df_model[col].fillna(df_model[col].median())

print("\nFeatures used for modelling:")
print(features)


# =========================
# 12. SAVE CLEAN FEATURE TABLE
# =========================

feature_table_cols = [
    "date",
    "suburb",
    "tmax",
    "tmin",
    "daily_temp_range",
    "night_start_time",
    "night_min_time",
    "night_start_temp",
    "night_min_temp",
    "night_hours_to_min",
    "night_total_hours",
    "night_cooling_rate",
    "hot_day",
    "cooling_failure",
    "cooling_risk_category",
]

for col in [
    "humidity_mean",
    "wind_speed_mean",
    "air_pressure_mean",
    "precipitation_total",
]:
    if col in df_model.columns:
        feature_table_cols.append(col)

feature_table_cols += ["month", "is_summer"]

feature_table = df_model[feature_table_cols].copy()

feature_table.to_csv(FEATURE_TABLE_FILE, index=False)

print("\nSaved cleaned feature table:")
print(FEATURE_TABLE_FILE)


# =========================
# 13. PREPARE ML DATA
# =========================

X = df_model[features]
y = df_model["cooling_failure"]

if y.nunique() < 2:
    raise ValueError(
        "Target has only one class. Model cannot train. "
        "Adjust hot-day threshold or cooling-failure threshold."
    )


# =========================
# 14. TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)


# =========================
# 15. BASELINE MODEL: LOGISTIC REGRESSION
# =========================

log_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
)

log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)

print("\n==============================")
print("LOGISTIC REGRESSION RESULTS")
print("==============================")
print(classification_report(y_test, log_pred, zero_division=0))


# =========================
# 16. MAIN MODEL: RANDOM FOREST
# =========================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    max_depth=4,
    min_samples_leaf=3,
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\n==============================")
print("RANDOM FOREST RESULTS")
print("==============================")
print(classification_report(y_test, rf_pred, zero_division=0))


# =========================
# 17. CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_test, rf_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No cooling failure", "Cooling failure"],
)

disp.plot()
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_FIG)
plt.close()

print("\nSaved confusion matrix:")
print(CONFUSION_MATRIX_FIG)


# =========================
# 18. FEATURE IMPORTANCE
# =========================

importance = pd.DataFrame(
    {
        "feature": features,
        "importance": rf_model.feature_importances_,
    }
)

importance = importance.sort_values(by="importance", ascending=False)

print("\nFeature importance:")
print(importance)

importance.to_csv(FEATURE_IMPORTANCE_FILE, index=False)

plt.figure(figsize=(8, 5))
plt.barh(importance["feature"], importance["importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(FEATURE_IMPORTANCE_FIG)
plt.close()

print("\nSaved feature importance:")
print(FEATURE_IMPORTANCE_FILE)
print(FEATURE_IMPORTANCE_FIG)


# =========================
# 19. SAVE MODEL OUTPUTS
# =========================

results = X_test.copy()
results["actual"] = y_test.values
results["predicted"] = rf_pred
results["prob_cooling_failure"] = rf_model.predict_proba(X_test)[:, 1]

results.to_csv(MODEL_PREDICTIONS_FILE, index=False)

print("\nSaved predictions:")
print(MODEL_PREDICTIONS_FILE)


# =========================
# 20. FINAL SUMMARY
# =========================

print("\n==============================")
print("MVP PIPELINE WITH METEOSTAT NIGHT-TIME COOLING RATE COMPLETE")
print("==============================")
print("Files created:")
print(f"- {FEATURE_TABLE_FILE}")
print(f"- {FEATURE_IMPORTANCE_FILE}")
print(f"- {MODEL_PREDICTIONS_FILE}")
print(f"- {CONFUSION_MATRIX_FIG}")
print(f"- {FEATURE_IMPORTANCE_FIG}")

print("\nTarget definition:")
print(
    "Cooling failure = hot day where overnight cooling rate is in the "
    "bottom 25% of hot-day cooling rates."
)

print("\nNight-time cooling-rate formula:")
print(
    "night_cooling_rate = "
    "(night_start_temp - night_min_temp) / hours from 18:00 to overnight minimum"
)

print("\nWeather source:")
print(
    "Temperature, humidity, wind speed, air pressure and precipitation are all "
    "taken from the same Meteostat hourly file."
)

print("\nFeatures used:")
print(features)