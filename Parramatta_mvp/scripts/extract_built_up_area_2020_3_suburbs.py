from pathlib import Path
import zipfile
import pandas as pd
import geopandas as gpd
import fiona


# =========================
# 0. PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data_raw"
BOUNDARIES_DIR = RAW_DIR / "boundaries"
BUILT_UP_DIR = RAW_DIR / "built_up_area"
PROCESSED_DIR = BASE_DIR / "data_processed"

SA2_ZIP = BOUNDARIES_DIR / "SA2_2021_AUST_SHP_GDA2020.zip"
SA2_UNZIPPED_DIR = BOUNDARIES_DIR / "SA2_2021_AUST_SHP_GDA2020"

BUILT_UP_FILE = BUILT_UP_DIR / "Built_Up_Areas.gpkg"

STATIC_FEATURES_FILE = PROCESSED_DIR / "suburb_static_features.csv"
BUILT_UP_OUTPUT_FILE = PROCESSED_DIR / "suburb_built_up_area_2020.csv"

PROCESSED_DIR.mkdir(exist_ok=True)


# =========================
# 1. CHECK FILES
# =========================

if not SA2_ZIP.exists():
    raise FileNotFoundError(f"Missing SA2 boundary zip: {SA2_ZIP}")

if not BUILT_UP_FILE.exists():
    raise FileNotFoundError(f"Missing built-up area file: {BUILT_UP_FILE}")

if not STATIC_FEATURES_FILE.exists():
    raise FileNotFoundError(f"Missing static features file: {STATIC_FEATURES_FILE}")


# =========================
# 2. UNZIP SA2 BOUNDARIES
# =========================

SA2_UNZIPPED_DIR.mkdir(exist_ok=True)

print("Unzipping SA2 boundary file if needed...")
with zipfile.ZipFile(SA2_ZIP, "r") as zip_ref:
    zip_ref.extractall(SA2_UNZIPPED_DIR)

shp_files = list(SA2_UNZIPPED_DIR.rglob("*.shp"))

if not shp_files:
    raise FileNotFoundError(f"No .shp files found after unzipping: {SA2_UNZIPPED_DIR}")

print("\nFound shapefiles:")
for shp in shp_files:
    print("-", shp)

SA2_SHP = shp_files[0]

print("\nUsing SA2 shapefile:")
print(SA2_SHP)


# =========================
# 3. LOAD SA2 BOUNDARIES
# =========================

sa2 = gpd.read_file(SA2_SHP)

print("\nSA2 columns:")
print(sa2.columns.tolist())

print("\nSA2 CRS:")
print(sa2.crs)

name_col = "SA2_NAME21"

if name_col not in sa2.columns:
    raise ValueError(f"Expected SA2 name column not found: {name_col}")

print(f"\nUsing SA2 name column: {name_col}")


# =========================
# 4. SELECT SA2 PROXY AREAS
# =========================

# Notes:
# - Bondi and Campbelltown match directly.
# - The older population SA2 proxy "Parramatta - Rosehill" is represented
#   in 2021 boundaries by Parramatta - North, Parramatta - South,
#   and Rosehill - Harris Park.
# - Their combined area is approximately 8.5 km2, matching the old population area.

suburb_to_sa2_list = {
    "Bondi": [
        "Bondi Beach - North Bondi",
    ],
    "Parramatta": [
        "Parramatta - North",
        "Parramatta - South",
        "Rosehill - Harris Park",
    ],
    "Campbelltown": [
        "Campbelltown - Woodbine",
    ],
}

all_selected_sa2_names = [
    sa2_name
    for names in suburb_to_sa2_list.values()
    for sa2_name in names
]

sa2_selected = sa2[sa2[name_col].isin(all_selected_sa2_names)].copy()

print("\nSelected SA2 areas:")
print(
    sa2_selected[
        ["SA2_CODE21", "SA2_NAME21", "SA3_NAME21", "SA4_NAME21", "AREASQKM21"]
    ].drop_duplicates().to_string(index=False)
)

missing = sorted(set(all_selected_sa2_names) - set(sa2_selected[name_col]))
if missing:
    raise ValueError(f"Could not find these SA2 names in boundary file: {missing}")

# Add suburb label.
sa2_name_to_suburb = {}
for suburb, names in suburb_to_sa2_list.items():
    for sa2_name in names:
        sa2_name_to_suburb[sa2_name] = suburb

sa2_selected["suburb"] = sa2_selected[name_col].map(sa2_name_to_suburb)

# Keep SA2 names used for reporting.
sa2_used = (
    sa2_selected
    .groupby("suburb")[name_col]
    .apply(lambda x: "; ".join(sorted(x.astype(str).unique())))
    .reset_index()
    .rename(columns={name_col: "sa2_names_used_for_built_up"})
)

# Project to EPSG:3857 to match built-up dataset and get square metres.
sa2_selected = sa2_selected.to_crs("EPSG:3857")
sa2_selected["sa2_part_area_m2"] = sa2_selected.geometry.area

# Dissolve multiple SA2s into one suburb geometry.
sa2_dissolved = (
    sa2_selected
    .dissolve(by="suburb", as_index=False)
)

sa2_dissolved["sa2_area_m2"] = sa2_dissolved.geometry.area

sa2_dissolved = sa2_dissolved.merge(
    sa2_used,
    on="suburb",
    how="left",
)

print("\nDissolved suburb proxy areas:")
print(
    sa2_dissolved[
        ["suburb", "sa2_names_used_for_built_up", "sa2_area_m2"]
    ].to_string(index=False)
)


# =========================
# 5. LOAD BUILT-UP AREAS
# =========================

layers = fiona.listlayers(BUILT_UP_FILE)
print("\nBuilt-up area layers:")
print(layers)

built_layer = "BuiltUpAreas_Source" if "BuiltUpAreas_Source" in layers else layers[0]

print(f"\nUsing built-up layer: {built_layer}")

built = gpd.read_file(BUILT_UP_FILE, layer=built_layer)

print("\nBuilt-up CRS:")
print(built.crs)

print("\nBuilt-up columns:")
print(built.columns.tolist())

if built.crs is None:
    raise ValueError("Built-up area file has no CRS.")

built = built.to_crs("EPSG:3857")

# Keep only valid geometries.
built = built[~built.geometry.isna()].copy()
built = built[built.geometry.is_valid].copy()

# Keep only built-up features.
if "feature_type" in built.columns:
    built = built[
        built["feature_type"].astype(str).str.lower().str.contains("built", na=False)
    ].copy()

print("\nBuilt-up rows after cleaning:")
print(len(built))


# =========================
# 6. SPATIAL INTERSECTION
# =========================

print("\nIntersecting built-up polygons with selected suburb proxy boundaries...")

intersection = gpd.overlay(
    built,
    sa2_dissolved[
        [
            "suburb",
            "sa2_names_used_for_built_up",
            "sa2_area_m2",
            "geometry",
        ]
    ],
    how="intersection",
)

if intersection.empty:
    raise ValueError(
        "Spatial intersection returned no rows. CRS or boundary selection may be wrong."
    )

intersection["intersect_area_m2"] = intersection.geometry.area

built_summary = (
    intersection
    .groupby(
        [
            "suburb",
            "sa2_names_used_for_built_up",
            "sa2_area_m2",
        ],
        as_index=False,
    )
    .agg(
        built_up_area_m2=("intersect_area_m2", "sum")
    )
)

built_summary["built_up_area_percent"] = (
    built_summary["built_up_area_m2"] / built_summary["sa2_area_m2"] * 100
)

built_summary["built_up_density_source_note"] = (
    "Digital Atlas Built Up Areas / Bing Building Footprints October 2020; "
    "intersected with ABS SA2 2021 GDA2020 boundaries. "
    "Parramatta uses combined 2021 SA2s: Parramatta - North, Parramatta - South, "
    "and Rosehill - Harris Park to match the older Parramatta - Rosehill proxy area."
)

# Nice ordering.
suburb_order = ["Bondi", "Parramatta", "Campbelltown"]
built_summary["suburb"] = pd.Categorical(
    built_summary["suburb"],
    categories=suburb_order,
    ordered=True,
)
built_summary = built_summary.sort_values("suburb").reset_index(drop=True)
built_summary["suburb"] = built_summary["suburb"].astype(str)

print("\nBuilt-up area summary:")
print(built_summary.to_string(index=False))

built_summary.to_csv(BUILT_UP_OUTPUT_FILE, index=False)

print("\nSaved built-up output:")
print(BUILT_UP_OUTPUT_FILE)


# =========================
# 7. MERGE INTO STATIC FEATURES
# =========================

static = pd.read_csv(STATIC_FEATURES_FILE)

# Drop old built-up columns if rerunning.
old_cols = [
    "built_up_area_m2",
    "sa2_area_m2",
    "built_up_area_percent",
    "built_up_density_source_note",
    "sa2_names_used_for_built_up",
]

static = static.drop(columns=[c for c in old_cols if c in static.columns])

static_updated = static.merge(
    built_summary[
        [
            "suburb",
            "built_up_area_m2",
            "sa2_area_m2",
            "built_up_area_percent",
            "built_up_density_source_note",
            "sa2_names_used_for_built_up",
        ]
    ],
    on="suburb",
    how="left",
)

backup_file = PROCESSED_DIR / "suburb_static_features_before_built_up_area.csv"
static.to_csv(backup_file, index=False)

static_updated.to_csv(STATIC_FEATURES_FILE, index=False)

print("\nUpdated static features file:")
print(STATIC_FEATURES_FILE)

print("\nBackup saved:")
print(backup_file)

print("\nUpdated static features preview:")
print(static_updated.to_string(index=False))

print("\nDone.")
