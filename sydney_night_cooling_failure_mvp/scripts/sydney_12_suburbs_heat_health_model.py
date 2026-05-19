from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# =========================
# 0. PROJECT PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data_processed"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
TABLES_DIR = BASE_DIR / "outputs" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_TABLE_FILE = PROCESSED_DIR / "sydney_12_suburbs_feature_table_night_cooling.csv"

HEALTH_METRICS_FILE = TABLES_DIR / "night_heat_health_risk_metrics_summary.csv"
HEALTH_PREDICTIONS_FILE = TABLES_DIR / "night_heat_health_risk_predictions.csv"
HEALTH_SUBURB_SUMMARY_FILE = TABLES_DIR / "night_heat_health_risk_suburb_summary.csv"
HEALTH_FEATURE_IMPORTANCE_FILE = TABLES_DIR / "night_heat_health_risk_feature_importance.csv"

HEALTH_CONFUSION_MATRIX_FIG = FIGURES_DIR / "night_heat_health_risk_confusion_matrix.png"
HEALTH_FEATURE_IMPORTANCE_FIG = FIGURES_DIR / "night_heat_health_risk_feature_importance.png"


# =========================
# 1. LOAD EXISTING FEATURE TABLE
# =========================

if not FEATURE_TABLE_FILE.exists():
    raise FileNotFoundError(
        f"Missing feature table: {FEATURE_TABLE_FILE}\n"
        "Run scripts/sydney_12_suburbs_model.py first."
    )

df = pd.read_csv(FEATURE_TABLE_FILE)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

required_cols = [
    "suburb",
    "date",
    "night_min_temp",
    "night_cooling_rate",
    "cooling_failure",
]

missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns from feature table: {missing}")


# =========================
# 2. CREATE NIGHT-TIME HEAT-HEALTH RISK TARGET
# =========================

# Health-risk refinement:
# A night is labelled as heat-health risk only if:
# 1. It was already a slow-cooling night.
# 2. The overnight minimum temperature stayed at or above 20°C.
#
# This avoids treating mild coastal nights as high health risk just because they cool slowly.
NIGHT_MIN_HEALTH_THRESHOLD = 20.0

df["night_heat_health_risk"] = (
    (df["cooling_failure"] == 1)
    & (df["night_min_temp"] >= NIGHT_MIN_HEALTH_THRESHOLD)
).astype(int)

print("\n==============================")
print("NIGHT-TIME HEAT-HEALTH RISK TARGET")
print("==============================")
print(f"Night minimum temperature threshold: {NIGHT_MIN_HEALTH_THRESHOLD:.1f}°C")
print("\nTarget definition:")
print(
    "night_heat_health_risk = 1 when a suburb-night cooled slowly "
    "AND the overnight minimum temperature stayed at or above 20°C."
)

print("\nTarget class counts:")
print(
    df["night_heat_health_risk"]
    .value_counts()
    .rename(index={0: "No night heat-health risk", 1: "Night heat-health risk"})
)

print(
    "\nPlain English: The original cooling-failure target only measured slow cooling. "
    "This new target is stricter. It only flags nights that cooled slowly and also stayed warm overnight. "
    "This is closer to the health-risk concern because warm nights reduce recovery from daytime heat."
)


# =========================
# 3. SUBURB-LEVEL HEALTH-RISK SUMMARY
# =========================

health_suburb_summary = (
    df.groupby("suburb")
    .agg(
        days=("date", "count"),
        hot_days=("hot_day", "sum") if "hot_day" in df.columns else ("date", "count"),
        cooling_failures=("cooling_failure", "sum"),
        night_heat_health_risk_days=("night_heat_health_risk", "sum"),
        mean_tmax=("tmax", "mean"),
        mean_night_min_temp=("night_min_temp", "mean"),
        mean_night_cooling_rate=("night_cooling_rate", "mean"),
        median_night_cooling_rate=("night_cooling_rate", "median"),
    )
    .reset_index()
)

health_suburb_summary["cooling_failure_rate_all_days"] = (
    health_suburb_summary["cooling_failures"] / health_suburb_summary["days"]
)

health_suburb_summary["night_heat_health_risk_rate_all_days"] = (
    health_suburb_summary["night_heat_health_risk_days"] / health_suburb_summary["days"]
)

health_suburb_summary.to_csv(HEALTH_SUBURB_SUMMARY_FILE, index=False)

display_summary = health_suburb_summary.copy()

for col in [
    "cooling_failure_rate_all_days",
    "night_heat_health_risk_rate_all_days",
]:
    display_summary[col] = (display_summary[col] * 100).round(1)

for col in [
    "mean_tmax",
    "mean_night_min_temp",
    "mean_night_cooling_rate",
    "median_night_cooling_rate",
]:
    display_summary[col] = display_summary[col].round(3)

display_summary = display_summary.sort_values(
    by="night_heat_health_risk_days",
    ascending=False,
)

print("\n==============================")
print("SUBURB-LEVEL NIGHT HEAT-HEALTH RISK SUMMARY")
print("==============================")
print(
    display_summary[
        [
            "suburb",
            "days",
            "hot_days",
            "cooling_failures",
            "night_heat_health_risk_days",
            "cooling_failure_rate_all_days",
            "night_heat_health_risk_rate_all_days",
            "mean_tmax",
            "mean_night_min_temp",
            "mean_night_cooling_rate",
        ]
    ].to_string(index=False)
)

print(
    "\nPlain English: This table separates slow cooling from actual warm-night risk. "
    "A suburb can cool slowly but still not be a strong heat-health concern if its nights are mild. "
    "The night_heat_health_risk_days column is the more health-relevant count."
)


# =========================
# 4. PREPARE FEATURES
# =========================

# Keep same defensible predictor set as final cooling-rate model.
# Do NOT use night_min_temp, tmin, daily_temp_range, night_cooling_rate,
# cooling_failure, or cooling_risk_category as predictors.
possible_features = [
    "tmax",
    "dewpoint_mean",
    "humidity_mean",
    "wind_speed_mean",
    "air_pressure_mean",
    "precipitation_total",
    "distance_to_coast_km",
    "distance_to_water_km",
    "is_coastal",
    "tree_canopy_percent",
    "population_density_people_per_km2",
    "month",
    "is_summer",
]

features = [col for col in possible_features if col in df.columns]

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[col].isna().all():
        print(f"Warning: dropping feature because it is entirely missing: {col}")
    else:
        df[col] = df[col].fillna(df[col].median())

features = [
    col for col in features
    if col in df.columns and not df[col].isna().all()
]

print("\nFeatures used for night heat-health risk model:")
print(features)

print("\nExcluded from modelling:")
print(
    [
        "night_min_temp",
        "tmin",
        "daily_temp_range",
        "night_cooling_rate",
        "cooling_failure",
        "cooling_risk_category",
        "built_up_area_percent",
    ]
)

print(
    "\nPlain English: The model is not allowed to use the actual overnight minimum temperature "
    "or cooling rate as input features, because those define the target. It must predict risk "
    "from broader weather, geography, tree canopy, population density, and season."
)


# =========================
# 5. TRAIN / TEST SPLIT
# =========================

X = df[features]
y = df["night_heat_health_risk"]

if y.nunique() < 2:
    raise ValueError(
        "night_heat_health_risk has only one class. "
        "Try lowering NIGHT_MIN_HEALTH_THRESHOLD from 20°C."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

print("\n==============================")
print("TRAIN / VALIDATION / TEST SPLIT")
print("==============================")
print(f"Total model rows: {len(df)}")
print(f"Training rows: {len(X_train)} ({len(X_train) / len(df):.1%})")
print(f"Testing rows: {len(X_test)} ({len(X_test) / len(df):.1%})")
print("Validation method: 5-fold cross-validation performed on the training set")

print("\nTraining target balance:")
print(
    y_train.value_counts()
    .rename(index={0: "No night heat-health risk", 1: "Night heat-health risk"})
)

print("\nTesting target balance:")
print(
    y_test.value_counts()
    .rename(index={0: "No night heat-health risk", 1: "Night heat-health risk"})
)


# =========================
# 6. LOGISTIC REGRESSION BASELINE
# =========================

log_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
    ),
)

log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

print("\n==============================")
print("LOGISTIC REGRESSION BASELINE - NIGHT HEAT-HEALTH RISK")
print("==============================")
print(classification_report(y_test, log_pred, zero_division=0))


# =========================
# 7. RANDOM FOREST MAIN MODEL
# =========================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    max_depth=4,
    min_samples_leaf=3,
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

cv_scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

cv_results = cross_validate(
    rf_model,
    X_train,
    y_train,
    cv=cv,
    scoring=cv_scoring,
    return_train_score=False,
)

cv_table = pd.DataFrame(
    {
        "fold": range(1, 6),
        "accuracy": cv_results["test_accuracy"],
        "precision": cv_results["test_precision"],
        "recall": cv_results["test_recall"],
        "f1": cv_results["test_f1"],
        "roc_auc": cv_results["test_roc_auc"],
    }
)

cv_summary = pd.DataFrame(
    {
        "fold": ["mean", "std"],
        "accuracy": [cv_table["accuracy"].mean(), cv_table["accuracy"].std()],
        "precision": [cv_table["precision"].mean(), cv_table["precision"].std()],
        "recall": [cv_table["recall"].mean(), cv_table["recall"].std()],
        "f1": [cv_table["f1"].mean(), cv_table["f1"].std()],
        "roc_auc": [cv_table["roc_auc"].mean(), cv_table["roc_auc"].std()],
    }
)

cv_output = pd.concat([cv_table, cv_summary], ignore_index=True)

print("\n==============================")
print("RANDOM FOREST 5-FOLD CROSS-VALIDATION - NIGHT HEAT-HEALTH RISK")
print("==============================")
print(cv_output)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("\n==============================")
print("RANDOM FOREST RESULTS - NIGHT HEAT-HEALTH RISK")
print("==============================")
print(classification_report(y_test, rf_pred, zero_division=0))

metrics_summary = pd.DataFrame(
    [
        {
            "model": "Logistic Regression baseline",
            "accuracy": accuracy_score(y_test, log_pred),
            "night_heat_health_precision": precision_score(y_test, log_pred, zero_division=0),
            "night_heat_health_recall": recall_score(y_test, log_pred, zero_division=0),
            "night_heat_health_f1": f1_score(y_test, log_pred, zero_division=0),
            "roc_auc": None,
        },
        {
            "model": "Random Forest final",
            "accuracy": accuracy_score(y_test, rf_pred),
            "night_heat_health_precision": precision_score(y_test, rf_pred, zero_division=0),
            "night_heat_health_recall": recall_score(y_test, rf_pred, zero_division=0),
            "night_heat_health_f1": f1_score(y_test, rf_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, rf_prob),
        },
        {
            "model": "Random Forest 5-fold CV mean",
            "accuracy": cv_table["accuracy"].mean(),
            "night_heat_health_precision": cv_table["precision"].mean(),
            "night_heat_health_recall": cv_table["recall"].mean(),
            "night_heat_health_f1": cv_table["f1"].mean(),
            "roc_auc": cv_table["roc_auc"].mean(),
        },
    ]
)

metrics_summary.to_csv(HEALTH_METRICS_FILE, index=False)

print("\nCompact health-risk metrics summary:")
print(metrics_summary)

print("\nSaved health-risk metrics summary:")
print(HEALTH_METRICS_FILE)


# =========================
# 8. CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_test, rf_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No night heat-health risk", "Night heat-health risk"],
)

disp.plot()
plt.title("Random Forest Confusion Matrix - Night Heat-Health Risk")
plt.tight_layout()
plt.savefig(HEALTH_CONFUSION_MATRIX_FIG)
plt.close()

tn, fp, fn, tp = cm.ravel()

print("\nConfusion matrix explained:")
print(f"- Correctly predicted no night heat-health risk: {tn}")
print(f"- False alarms, predicted heat-health risk but it was not: {fp}")
print(f"- Missed night heat-health risk cases: {fn}")
print(f"- Correctly detected night heat-health risk cases: {tp}")
print(
    "Plain English: For a health-risk warning model, missed heat-health risk nights are the most concerning number."
)

print("\nSaved health-risk confusion matrix:")
print(HEALTH_CONFUSION_MATRIX_FIG)


# =========================
# 9. FEATURE IMPORTANCE
# =========================

importance = pd.DataFrame(
    {
        "feature": features,
        "importance": rf_model.feature_importances_,
    }
).sort_values(by="importance", ascending=False)

importance.to_csv(HEALTH_FEATURE_IMPORTANCE_FILE, index=False)

print("\nFeature importance - night heat-health risk:")
print(importance)

print(
    "\nPlain English: These are the variables the model relied on most when predicting warm, slow-cooling nights. "
    "Feature importance helps explain patterns, but does not prove direct causation."
)

plt.figure(figsize=(8, 5))
plt.barh(importance["feature"], importance["importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance - Night Heat-Health Risk")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(HEALTH_FEATURE_IMPORTANCE_FIG)
plt.close()

print("\nSaved health-risk feature importance:")
print(HEALTH_FEATURE_IMPORTANCE_FILE)
print(HEALTH_FEATURE_IMPORTANCE_FIG)


# =========================
# 10. SAVE PREDICTIONS
# =========================

results = df.loc[
    X_test.index,
    [
        "date",
        "suburb",
        "night_heat_health_risk",
        "cooling_failure",
        "night_cooling_rate",
        "night_min_temp",
        "hot_day",
    ],
].copy()

results = results.reset_index(drop=True)

X_test_reset = X_test.reset_index(drop=True)
results = pd.concat([results, X_test_reset], axis=1)

results["actual"] = y_test.reset_index(drop=True)
results["predicted"] = rf_pred
results["prob_night_heat_health_risk"] = rf_prob

results.to_csv(HEALTH_PREDICTIONS_FILE, index=False)

print("\nSaved health-risk predictions:")
print(HEALTH_PREDICTIONS_FILE)


# =========================
# 11. FINAL SUMMARY
# =========================

print("\n==============================")
print("NIGHT-TIME HEAT-HEALTH RISK MODEL COMPLETE")
print("==============================")

print("\nFiles created:")
print(f"- {HEALTH_METRICS_FILE}")
print(f"- {HEALTH_PREDICTIONS_FILE}")
print(f"- {HEALTH_SUBURB_SUMMARY_FILE}")
print(f"- {HEALTH_FEATURE_IMPORTANCE_FILE}")
print(f"- {HEALTH_CONFUSION_MATRIX_FIG}")
print(f"- {HEALTH_FEATURE_IMPORTANCE_FIG}")

print("\nPlain-English interpretation:")
print(
    "The original cooling-failure model identifies nights where suburbs cooled slowly. "
    "This health-risk model is stricter: it identifies nights where the suburb cooled slowly "
    "and the overnight minimum temperature stayed at or above 20°C."
)
print(
    "This reduces the chance of labelling mild coastal nights as high health risk just because "
    "their temperatures changed slowly overnight."
)
print(
    "Use this model when discussing night-time heat-health risk. "
    "Use the original model when discussing cooling-rate behaviour."
)
