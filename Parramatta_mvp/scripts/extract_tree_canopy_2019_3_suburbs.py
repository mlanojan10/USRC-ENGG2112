from pathlib import Path

import pandas as pd


# =========================
# 0. PROJECT PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_FILE = BASE_DIR / "data_raw" / "tree_canopy" / "tree_canopy_suburb_2019.xlsx"
PROCESSED_DIR = BASE_DIR / "data_processed"

OUT_FILE = PROCESSED_DIR / "suburb_tree_canopy_2019.csv"
STATIC_FEATURES_FILE = PROCESSED_DIR / "suburb_static_features.csv"
UPDATED_STATIC_FEATURES_FILE = PROCESSED_DIR / "suburb_static_features_with_canopy.csv"

PROCESSED_DIR.mkdir(exist_ok=True)


# =========================
# 1. CHECK INPUT FILE
# =========================

if not RAW_FILE.exists():
    raise FileNotFoundError(f"Missing tree canopy Excel file: {RAW_FILE}")


# =========================
# 2. LOAD SUBURB TOTAL SHEET
# =========================

sheet_name = "2019 Suburb (Total)"

df = pd.read_excel(RAW_FILE, sheet_name=sheet_name)

print("Original columns:")
print(df.columns.tolist())

print("\nFirst rows:")
print(df.head())


# =========================
# 3. CLEAN COLUMN NAMES
# =========================

df = df.rename(
    columns={
        "Suburb Name": "suburb",
        "Suburb Area (inside GSR)": "suburb_area_inside_gsr_m2",
        "Cadastred Suburb Area": "cadastred_suburb_area_m2",
        "Canopy Area": "tree_canopy_area_m2",
        "% Canopy Cover": "tree_canopy_percent",
    }
)

required_cols = [
    "suburb",
    "suburb_area_inside_gsr_m2",
    "cadastred_suburb_area_m2",
    "tree_canopy_area_m2",
    "tree_canopy_percent",
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Missing expected columns after rename: {missing}")

df["suburb"] = df["suburb"].astype(str).str.strip().str.title()

for col in [
    "suburb_area_inside_gsr_m2",
    "cadastred_suburb_area_m2",
    "tree_canopy_area_m2",
    "tree_canopy_percent",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# =========================
# 4. FILTER TO THREE MVP SUBURBS
# =========================

target_suburbs = [
    "Bondi",
    "Campbelltown",
    "Parramatta",
]

tree_3 = df[df["suburb"].isin(target_suburbs)].copy()

print("\nExtracted 3-suburb canopy table:")
print(tree_3)

found_suburbs = set(tree_3["suburb"])
missing_suburbs = [s for s in target_suburbs if s not in found_suburbs]

if missing_suburbs:
    print("\nWARNING: These suburbs were not found exactly:")
    print(missing_suburbs)
    print("\nClosest matches containing the suburb text:")

    for suburb in missing_suburbs:
        matches = df[df["suburb"].str.contains(suburb, case=False, na=False)]
        print(f"\n{suburb}:")
        print(matches[["suburb", "tree_canopy_percent", "tree_canopy_area_m2"]].head(20))

    raise ValueError("Some target suburbs were not found. Check suburb spelling in Excel.")


# =========================
# 5. SAVE CANOPY-ONLY TABLE
# =========================

tree_3 = tree_3[
    [
        "suburb",
        "tree_canopy_percent",
        "tree_canopy_area_m2",
        "suburb_area_inside_gsr_m2",
        "cadastred_suburb_area_m2",
    ]
].copy()

tree_3.to_csv(OUT_FILE, index=False)

print("\nSaved canopy table:")
print(OUT_FILE)


# =========================
# 6. MERGE INTO STATIC FEATURE TABLE
# =========================

if not STATIC_FEATURES_FILE.exists():
    raise FileNotFoundError(f"Missing static feature file: {STATIC_FEATURES_FILE}")

static = pd.read_csv(STATIC_FEATURES_FILE)

print("\nOriginal static features:")
print(static)

if "suburb" not in static.columns:
    raise ValueError("Static feature table must contain a 'suburb' column.")

static["suburb"] = static["suburb"].astype(str).str.strip().str.title()

updated = static.merge(
    tree_3,
    on="suburb",
    how="left",
)

print("\nUpdated static features with canopy:")
print(updated)

if updated["tree_canopy_percent"].isna().any():
    print("\nWARNING: Some suburbs did not receive canopy data:")
    print(updated[updated["tree_canopy_percent"].isna()])

updated.to_csv(UPDATED_STATIC_FEATURES_FILE, index=False)

print("\nSaved updated static features:")
print(UPDATED_STATIC_FEATURES_FILE)


# =========================
# 7. OPTIONAL: OVERWRITE MAIN STATIC FEATURE FILE
# =========================

updated.to_csv(STATIC_FEATURES_FILE, index=False)

print("\nAlso overwritten main static feature file:")
print(STATIC_FEATURES_FILE)
