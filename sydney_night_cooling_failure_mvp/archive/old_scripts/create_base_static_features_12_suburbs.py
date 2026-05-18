from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

METADATA_FILE = BASE_DIR / "metadata" / "suburbs_12_mvp.csv"
PROCESSED_DIR = BASE_DIR / "data_processed"
OUTPUT_FILE = PROCESSED_DIR / "suburb_static_features_12_base.csv"

PROCESSED_DIR.mkdir(exist_ok=True)

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
    raise ValueError(f"Missing required columns: {missing}")

static = suburbs[
    [
        "suburb",
        "distance_to_coast_km",
        "distance_to_water_km",
        "is_coastal",
        "inland_category",
    ]
].copy()

static.to_csv(OUTPUT_FILE, index=False)

print("Saved 12-suburb base static features:")
print(OUTPUT_FILE)

print("\nPreview:")
print(static)
