import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# =========================
# 0. FILE PATHS
# =========================

RAW_DIR = "data_raw"
PROCESSED_DIR = "data_processed"
FIGURES_DIR = "outputs/figures"
TABLES_DIR = "outputs/tables"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

WEATHER_FILE = os.path.join(RAW_DIR, "parramatta_weather_2020.csv")
HUMIDITY_FILE = os.path.join(PROCESSED_DIR, "parramatta_humidity_2020.csv")
WIND_FILE = os.path.join(PROCESSED_DIR, "parramatta_wind_2020.csv")

FEATURE_TABLE_FILE = os.path.join(
    PROCESSED_DIR,
    "parramatta_feature_table_with_humidity_wind.csv"
)

FEATURE_IMPORTANCE_FILE = os.path.join(TABLES_DIR, "feature_importance.csv")
MODEL_PREDICTIONS_FILE = os.path.join(TABLES_DIR, "model_predictions.csv")

CONFUSION_MATRIX_FIG = os.path.join(FIGURES_DIR, "confusion_matrix.png")
FEATURE_IMPORTANCE_FIG = os.path.join(FIGURES_DIR, "feature_importance.png")


# =========================
# 1. LOAD DAILY WEATHER DATA
# =========================

df = pd.read_csv(WEATHER_FILE)

print("Original weather columns:")
print(df.columns)

print("\nFirst 5 weather rows:")
print(df.head())


# =========================
# 2. BASIC WEATHER CLEANING
# =========================

df = df[["date", "tmin", "tmax", "prcp"]].copy()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["tmin"] = pd.to_numeric(df["tmin"], errors="coerce")
df["tmax"] = pd.to_numeric(df["tmax"], errors="coerce")
df["prcp"] = pd.to_numeric(df["prcp"], errors="coerce")

df = df.dropna(subset=["date", "tmin", "tmax"])
df["prcp"] = df["prcp"].fillna(0)

df["suburb"] = "Parramatta"


# =========================
# 3. LOAD HUMIDITY DATA
# =========================

humidity = pd.read_csv(HUMIDITY_FILE)

print("\nOriginal humidity columns:")
print(humidity.columns)

print("\nFirst 5 humidity rows:")
print(humidity.head())

humidity["date"] = pd.to_datetime(humidity["date"], errors="coerce")
humidity["humidity_mean"] = pd.to_numeric(
    humidity["humidity_mean"],
    errors="coerce"
)

humidity = humidity.dropna(subset=["date", "humidity_mean"])


# =========================
# 4. MERGE HUMIDITY DATA
# =========================

df = df.merge(
    humidity[["date", "humidity_mean"]],
    on="date",
    how="left"
)

df["humidity_mean"] = df["humidity_mean"].fillna(
    df["humidity_mean"].median()
)

print("\nAfter humidity merge:")
print(df.head())

print("\nMissing humidity values:")
print(df["humidity_mean"].isna().sum())


# =========================
# 5. LOAD WIND SPEED DATA
# =========================

wind = pd.read_csv(WIND_FILE)

print("\nOriginal wind columns:")
print(wind.columns)

print("\nFirst 5 wind rows:")
print(wind.head())

wind["date"] = pd.to_datetime(wind["date"], errors="coerce")
wind["wind_speed_mean"] = pd.to_numeric(
    wind["wind_speed_mean"],
    errors="coerce"
)

wind = wind.dropna(subset=["date", "wind_speed_mean"])


# =========================
# 6. MERGE WIND SPEED DATA
# =========================

df = df.merge(
    wind[["date", "wind_speed_mean"]],
    on="date",
    how="left"
)

df["wind_speed_mean"] = df["wind_speed_mean"].fillna(
    df["wind_speed_mean"].median()
)

print("\nAfter wind merge:")
print(df.head())

print("\nMissing wind speed values:")
print(df["wind_speed_mean"].isna().sum())


# =========================
# 7. FEATURE ENGINEERING
# =========================

df["daily_temp_range"] = df["tmax"] - df["tmin"]

df["month"] = df["date"].dt.month

df["is_summer"] = df["month"].isin([12, 1, 2]).astype(int)

# Hot day = top 30% hottest days
hot_threshold = df["tmax"].quantile(0.70)
df["hot_day"] = (df["tmax"] >= hot_threshold).astype(int)

print("\nHot day threshold:")
print(hot_threshold)

print("\nHot day counts:")
print(df["hot_day"].value_counts())


# =========================
# 8. CREATE TARGET VARIABLE
# =========================

hot_days = df[df["hot_day"] == 1].copy()

if len(hot_days) < 10:
    raise ValueError(
        "Not enough hot-day samples. Use more weather data or lower the hot-day threshold."
    )

# Poor cooling proxy:
# hot day where minimum temperature is in the top 40% of hot-day minimum temperatures
high_tmin_threshold = hot_days["tmin"].quantile(0.60)

df["poor_cooling_proxy"] = (
    (df["hot_day"] == 1) &
    (df["tmin"] >= high_tmin_threshold)
).astype(int)

print("\nHigh minimum temperature threshold among hot days:")
print(high_tmin_threshold)

print("\nTarget class counts:")
print(df["poor_cooling_proxy"].value_counts())


# =========================
# 9. SAVE CLEAN FEATURE TABLE
# =========================

feature_table = df[
    [
        "date",
        "suburb",
        "tmin",
        "tmax",
        "prcp",
        "humidity_mean",
        "wind_speed_mean",
        "daily_temp_range",
        "month",
        "is_summer",
        "hot_day",
        "poor_cooling_proxy",
    ]
].copy()

feature_table.to_csv(FEATURE_TABLE_FILE, index=False)

print("\nSaved cleaned feature table:")
print(FEATURE_TABLE_FILE)


# =========================
# 10. PREPARE ML DATA
# =========================

# Exclude tmin and hot_day to reduce label leakage.
features = [
    "tmax",
    "prcp",
    "humidity_mean",
    "wind_speed_mean",
    "daily_temp_range",
    "month",
    "is_summer",
]

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

X = df[features]
y = df["poor_cooling_proxy"]

if y.nunique() < 2:
    raise ValueError("Target has only one class. Model cannot train. Adjust threshold.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# =========================
# 11. BASELINE MODEL: LOGISTIC REGRESSION
# =========================

log_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)

print("\n==============================")
print("LOGISTIC REGRESSION RESULTS")
print("==============================")
print(classification_report(y_test, log_pred, zero_division=0))


# =========================
# 12. MAIN MODEL: RANDOM FOREST
# =========================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    max_depth=4,
    min_samples_leaf=3
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\n==============================")
print("RANDOM FOREST RESULTS")
print("==============================")
print(classification_report(y_test, rf_pred, zero_division=0))


# =========================
# 13. CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_test, rf_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No poor cooling", "Poor cooling"]
)

disp.plot()
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_FIG)
plt.close()

print("\nSaved confusion matrix:")
print(CONFUSION_MATRIX_FIG)


# =========================
# 14. FEATURE IMPORTANCE
# =========================

importance = pd.DataFrame({
    "feature": features,
    "importance": rf_model.feature_importances_
})

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
# 15. SAVE MODEL OUTPUTS
# =========================

results = X_test.copy()
results["actual"] = y_test.values
results["predicted"] = rf_pred
results["prob_poor_cooling"] = rf_model.predict_proba(X_test)[:, 1]

results.to_csv(MODEL_PREDICTIONS_FILE, index=False)

print("\nSaved predictions:")
print(MODEL_PREDICTIONS_FILE)


# =========================
# 16. FINAL SUMMARY
# =========================

print("\n==============================")
print("MVP PIPELINE WITH HUMIDITY + WIND COMPLETE")
print("==============================")
print("Files created:")
print(f"- {FEATURE_TABLE_FILE}")
print(f"- {FEATURE_IMPORTANCE_FILE}")
print(f"- {MODEL_PREDICTIONS_FILE}")
print(f"- {CONFUSION_MATRIX_FIG}")
print(f"- {FEATURE_IMPORTANCE_FIG}")