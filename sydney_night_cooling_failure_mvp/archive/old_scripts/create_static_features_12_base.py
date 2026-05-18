from pathlib import Path
import pandas as pd


# =========================
# 0. PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

METADATA_FILE = BASE_DIR / "metadata" / "suburbs_12_mvp.csv"
PROCESSED_DIR = BASE_DIR / "data_processed"

STATIC_FEATURES_FILE = PROCESSED_DIR / "suburb_static_features.csv"
BACKUP_FILE = PROCESSED_DIR / "suburb_static_features_3_suburbs_backup_before_12.csv"

PROCESSED_DIR.mkdir(exist_ok=True)


# =========================
# 1. LOAD METADATA
# =========================

if not METADATA_FILE.exists():
    raise FileNotFoundError(f"Missing metadata file: {METADATA_FILE}")

suburbs = pd.read_csv(METADATA_FILE)

required_cols = [
    "suburb",
    "suburb_lat",
    "suburb_lon",
    "inland_category",
    "is_coastal",
    "distance_to_coast_km",
    "distance_to_water_km",
]

missing = [c for c in required_cols if c not in suburbs.columns]

if missing:
    raise ValueError(f"Missing required columns in metadata file: {missing}")


# =========================
# 2. BACKUP EXISTING STATIC FILE
# =========================

if STATIC_FEATURES_FILE.exists():
    old_static = pd.read_csv(STATIC_FEATURES_FILE)
    old_static.to_csv(BACKUP_FILE, index=False)

    print("Backed up old static feature file to:")
    print(BACKUP_FILE)


# =========================
# 3. CREATE 12-SUBURB BASE STATIC FEATURES
# =========================

static = suburbs[
    [
        "suburb",
        "distance_to_coast_km",
        "distance_to_water_km",
        "is_coastal",
        "inland_category",
    ]
].copy()

suburb_order = [
    "Bondi",
    "Manly",
    "Coogee",
    "Cronulla",
    "Parramatta",
    "Ryde",
    "Bankstown",
    "Fairfield",
    "Penrith",
    "Blacktown",
    "Liverpool",
    "Campbelltown",
]

static["suburb"] = pd.Categorical(
    static["suburb"],
    categories=suburb_order,
    ordered=True,
)

static = static.sort_values("suburb").reset_index(drop=True)
static["suburb"] = static["suburb"].astype(str)

numeric_cols = [
    "distance_to_coast_km",
    "distance_to_water_km",
    "is_coastal",
]

for col in numeric_cols:
    static[col] = pd.to_numeric(static[col], errors="coerce")

if static[numeric_cols].isna().any().any():
    print("Warning: missing numeric values:")
    print(static[static[numeric_cols].isna().any(axis=1)])


# =========================
# 4. SAVE
# =========================

static.to_csv(STATIC_FEATURES_FILE, index=False)

print("\nSaved 12-suburb static feature file:")
print(STATIC_FEATURES_FILE)

print("\nStatic feature table:")
print(static.to_string(index=False))

print("\nRows:")
print(len(static))

print("\nColumns:")
print(static.columns.tolist())

print("\nDone.")
