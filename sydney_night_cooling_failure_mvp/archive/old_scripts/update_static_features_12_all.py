from pathlib import Path
import zipfile
import difflib

import pandas as pd
import geopandas as gpd
import fiona


# =========================
# 0. PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

METADATA_FILE = BASE_DIR / "metadata" / "suburbs_12_mvp.csv"

RAW_DIR = BASE_DIR / "data_raw"
TREE_FILE = RAW_DIR / "tree_canopy" / "tree_canopy_suburb_2019.xlsx"
POP_FILE = RAW_DIR / "population" / "sa2_population_2019_2020.xls"

BOUNDARIES_DIR = RAW_DIR / "boundaries"
SA2_ZIP = BOUNDARIES_DIR / "SA2_2021_AUST_SHP_GDA2020.zip"
SA2_UNZIPPED_DIR = BOUNDARIES_DIR / "SA2_2021_AUST_SHP_GDA2020"

BUILT_UP_FILE = RAW_DIR / "built_up_area" / "Built_Up_Areas.gpkg"

PROCESSED_DIR = BASE_DIR / "data_processed"
STATIC_FILE = PROCESSED_DIR / "suburb_static_features.csv"

TREE_OUTPUT = PROCESSED_DIR / "suburb_tree_canopy_2019_12.csv"
POP_OUTPUT = PROCESSED_DIR / "suburb_population_density_2020_12.csv"
BUILT_OUTPUT = PROCESSED_DIR / "suburb_built_up_area_2020_12.csv"

BACKUP_FILE = PROCESSED_DIR / "suburb_static_features_backup_before_12_full_update.csv"

PROCESSED_DIR.mkdir(exist_ok=True)


# =========================
# 1. LOAD 12-SUBURB BASE STATIC FEATURES
# =========================

if not METADATA_FILE.exists():
    raise FileNotFoundError(f"Missing metadata file: {METADATA_FILE}")

if not STATIC_FILE.exists():
    raise FileNotFoundError(f"Missing static feature file: {STATIC_FILE}")

metadata = pd.read_csv(METADATA_FILE)
static = pd.read_csv(STATIC_FILE)

static.to_csv(BACKUP_FILE, index=False)

print("Backed up current static feature file to:")
print(BACKUP_FILE)

required_static_cols = [
    "suburb",
    "distance_to_coast_km",
    "distance_to_water_km",
    "is_coastal",
    "inland_category",
]

missing = [c for c in required_static_cols if c not in static.columns]
if missing:
    raise ValueError(f"Static feature file is missing base columns: {missing}")

suburb_order = metadata["suburb"].tolist()

print("\nCurrent suburbs:")
print(suburb_order)


# =========================
# 2. ADD TREE CANOPY
# =========================

print("\n" + "=" * 70)
print("ADDING TREE CANOPY")
print("=" * 70)

if not TREE_FILE.exists():
    raise FileNotFoundError(f"Missing tree canopy file: {TREE_FILE}")

tree = pd.read_excel(TREE_FILE, sheet_name="2019 Suburb (Total)")

print("\nTree canopy columns:")
print(tree.columns.tolist())

tree_required = [
    "Suburb Name",
    "Suburb Area (inside GSR)",
    "Cadastred Suburb Area",
    "Canopy Area",
    "% Canopy Cover",
]

missing_tree = [c for c in tree_required if c not in tree.columns]
if missing_tree:
    raise ValueError(f"Tree canopy file missing columns: {missing_tree}")

tree_clean = tree[tree_required].copy()
tree_clean["suburb_key"] = tree_clean["Suburb Name"].astype(str).str.strip().str.upper()

tree_rows = []

for suburb in suburb_order:
    key = suburb.strip().upper()
    match = tree_clean[tree_clean["suburb_key"] == key]

    if match.empty:
        possible = tree_clean[
            tree_clean["suburb_key"].str.contains(key, na=False)
            | tree_clean["suburb_key"].apply(lambda x: key in x or x in key)
        ][["Suburb Name", "% Canopy Cover"]].head(10)

        print(f"\nWARNING: no direct tree canopy match for {suburb}. Possible matches:")
        print(possible.to_string(index=False))

        tree_rows.append(
            {
                "suburb": suburb,
                "tree_canopy_percent": pd.NA,
                "tree_canopy_area_m2": pd.NA,
                "suburb_area_inside_gsr_m2": pd.NA,
                "cadastred_suburb_area_m2": pd.NA,
            }
        )
        continue

    row = match.iloc[0]

    tree_rows.append(
        {
            "suburb": suburb,
            "tree_canopy_percent": row["% Canopy Cover"],
            "tree_canopy_area_m2": row["Canopy Area"],
            "suburb_area_inside_gsr_m2": row["Suburb Area (inside GSR)"],
            "cadastred_suburb_area_m2": row["Cadastred Suburb Area"],
        }
    )

tree_out = pd.DataFrame(tree_rows)
tree_out.to_csv(TREE_OUTPUT, index=False)

print("\nSaved tree canopy output:")
print(TREE_OUTPUT)
print(tree_out.to_string(index=False))


# =========================
# 3. ADD POPULATION DENSITY
# =========================

print("\n" + "=" * 70)
print("ADDING POPULATION DENSITY")
print("=" * 70)

if not POP_FILE.exists():
    raise FileNotFoundError(f"Missing population file: {POP_FILE}")

# Manual SA2 proxy map for ABS 2019-20 population file.
# These are suburb-level proxy choices. If any name fails, the script prints candidates.
population_sa2_map = {
    "Bondi": "Bondi Beach - North Bondi",
    "Manly": "Manly - Fairlight",
    "Coogee": "Coogee - Clovelly",
    "Cronulla": "Cronulla - Kurnell - Bundeena",
    "Parramatta": "Parramatta - Rosehill",
    "Ryde": "Ryde",
    "Bankstown": "Bankstown",
    "Fairfield": "Fairfield",
    "Penrith": "Penrith",
    "Blacktown": "Blacktown",
    "Liverpool": "Liverpool",
    "Campbelltown": "Campbelltown - Woodbine",
}


def load_population_table(pop_file):
    raw = pd.read_excel(pop_file, sheet_name="Table 1", header=None)

    # Find a likely header row containing SA2 code/name.
    header_row = None
    for i in range(min(30, len(raw))):
        vals = raw.iloc[i].astype(str).str.lower().tolist()
        row_text = " | ".join(vals)
        if "sa2" in row_text and "name" in row_text and ("2020" in row_text or "area" in row_text):
            header_row = i
            break

    if header_row is None:
        # This was the most likely layout from your earlier working script.
        header_row = 6

    df = pd.read_excel(pop_file, sheet_name="Table 1", header=header_row)

    # Drop empty columns.
    df = df.dropna(axis=1, how="all")

    # Standardise column names.
    original_cols = df.columns.tolist()
    lower_cols = [str(c).strip().lower() for c in original_cols]

    def find_col(possible_patterns):
        for pattern in possible_patterns:
            for original, lower in zip(original_cols, lower_cols):
                if pattern in lower:
                    return original
        return None

    sa2_code_col = find_col(["sa2 code", "sa2_code"])
    sa2_name_col = find_col(["sa2 name", "sa2_name"])

    # Try to identify population, area, density columns.
    pop_col = None
    area_col = None
    density_col = None

    for original, lower in zip(original_cols, lower_cols):
        if "2020" in lower and ("erp" in lower or "population" in lower or "persons" in lower):
            if "density" not in lower and pop_col is None:
                pop_col = original

        if ("area" in lower and ("km" in lower or "sq" in lower)) and area_col is None:
            area_col = original

        if "density" in lower and density_col is None:
            density_col = original

    # Fallback by known positions if column names are messy.
    if sa2_code_col is None or sa2_name_col is None:
        print("\nPopulation table columns:")
        print(original_cols)
        raise ValueError("Could not identify SA2 code/name columns in population file.")

    if pop_col is None or area_col is None or density_col is None:
        print("\nPopulation table columns:")
        print(original_cols)

        # Try numeric fallback based on earlier cleaned table structure.
        # Keep only rows with valid SA2 code/name, then infer.
        temp = df.copy()
        print("\nCould not confidently identify population/area/density columns.")
        print("The script will try a fallback from likely ABS Regional Population layout.")

        # The previous working extraction produced:
        # sa2_code, sa2_name, population_2020, area_km2, density.
        # For many ABS Regional Population files, useful columns are near the end.
        numeric_cols = []
        for col in df.columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 50:
                numeric_cols.append(col)

        print("\nNumeric-like columns:")
        print(numeric_cols)

        # Use last few numeric columns as fallback.
        # This is intentionally guarded below by sanity checks.
        if len(numeric_cols) >= 3:
            pop_col = numeric_cols[-3]
            area_col = numeric_cols[-2]
            density_col = numeric_cols[-1]
        else:
            raise ValueError("Could not identify population/area/density columns.")

    cleaned = df[[sa2_code_col, sa2_name_col, pop_col, area_col, density_col]].copy()
    cleaned.columns = [
        "sa2_code",
        "sa2_name",
        "population_2020",
        "area_km2",
        "population_density_people_per_km2",
    ]

    cleaned = cleaned.dropna(subset=["sa2_name"]).copy()
    cleaned["sa2_name"] = cleaned["sa2_name"].astype(str).str.strip()

    cleaned["population_2020"] = pd.to_numeric(cleaned["population_2020"], errors="coerce")
    cleaned["area_km2"] = pd.to_numeric(cleaned["area_km2"], errors="coerce")
    cleaned["population_density_people_per_km2"] = pd.to_numeric(
        cleaned["population_density_people_per_km2"],
        errors="coerce",
    )

    cleaned = cleaned.dropna(
        subset=[
            "population_2020",
            "area_km2",
            "population_density_people_per_km2",
        ]
    ).copy()

    # Remove non-SA2 rows by requiring a numeric-ish SA2 code.
    cleaned["sa2_code"] = cleaned["sa2_code"].astype(str).str.extract(r"(\d+)")[0]
    cleaned = cleaned.dropna(subset=["sa2_code"]).copy()

    return cleaned


pop_table = load_population_table(POP_FILE)

print("\nPopulation table preview:")
print(pop_table.head().to_string(index=False))

pop_rows = []

for suburb in suburb_order:
    target_sa2 = population_sa2_map.get(suburb)

    exact = pop_table[pop_table["sa2_name"] == target_sa2]

    if exact.empty:
        # Try case-insensitive.
        exact = pop_table[
            pop_table["sa2_name"].str.lower() == str(target_sa2).lower()
        ]

    if exact.empty:
        # Print candidates and use closest string if available.
        candidates = pop_table[
            pop_table["sa2_name"].str.contains(suburb, case=False, na=False)
        ][
            [
                "sa2_code",
                "sa2_name",
                "population_2020",
                "area_km2",
                "population_density_people_per_km2",
            ]
        ].head(20)

        close = difflib.get_close_matches(
            target_sa2,
            pop_table["sa2_name"].tolist(),
            n=5,
            cutoff=0.6,
        )

        print(f"\nWARNING: no exact population SA2 match for {suburb}: {target_sa2}")
        print("Candidates containing suburb name:")
        print(candidates.to_string(index=False))
        print("Close string matches:")
        print(close)

        if close:
            exact = pop_table[pop_table["sa2_name"] == close[0]]
            print(f"Using closest match for {suburb}: {close[0]}")
        else:
            pop_rows.append(
                {
                    "suburb": suburb,
                    "sa2_name": target_sa2,
                    "population_2020": pd.NA,
                    "population_density_people_per_km2": pd.NA,
                    "area_km2": pd.NA,
                    "population_density_source_note": "ABS Regional Population 2019-20, SA2 proxy; no match found",
                }
            )
            continue

    row = exact.iloc[0]

    pop_rows.append(
        {
            "suburb": suburb,
            "sa2_name": row["sa2_name"],
            "population_2020": row["population_2020"],
            "population_density_people_per_km2": row["population_density_people_per_km2"],
            "area_km2": row["area_km2"],
            "population_density_source_note": "ABS Regional Population 2019-20, SA2 proxy",
        }
    )

pop_out = pd.DataFrame(pop_rows)
pop_out.to_csv(POP_OUTPUT, index=False)

print("\nSaved population output:")
print(POP_OUTPUT)
print(pop_out.to_string(index=False))

if pop_out["population_density_people_per_km2"].isna().any():
    print("\nWARNING: Some population density values are missing. Check the warnings above.")


# =========================
# 4. ADD BUILT-UP AREA PERCENT
# =========================

print("\n" + "=" * 70)
print("ADDING BUILT-UP AREA PERCENT")
print("=" * 70)

if not SA2_ZIP.exists():
    raise FileNotFoundError(f"Missing SA2 boundary zip: {SA2_ZIP}")

if not BUILT_UP_FILE.exists():
    raise FileNotFoundError(f"Missing built-up file: {BUILT_UP_FILE}")

SA2_UNZIPPED_DIR.mkdir(exist_ok=True)

with zipfile.ZipFile(SA2_ZIP, "r") as zip_ref:
    zip_ref.extractall(SA2_UNZIPPED_DIR)

shp_files = list(SA2_UNZIPPED_DIR.rglob("*.shp"))
if not shp_files:
    raise FileNotFoundError("No SA2 shapefile found after unzipping.")

sa2 = gpd.read_file(shp_files[0])

if "SA2_NAME21" not in sa2.columns:
    raise ValueError("Expected SA2_NAME21 column not found in SA2 boundary file.")

# Use point-in-SA2 matching from the metadata coordinates.
points = gpd.GeoDataFrame(
    metadata[["suburb", "suburb_lat", "suburb_lon"]].copy(),
    geometry=gpd.points_from_xy(metadata["suburb_lon"], metadata["suburb_lat"]),
    crs="EPSG:4326",
)

points = points.to_crs(sa2.crs)

matched = gpd.sjoin(
    points,
    sa2[["SA2_CODE21", "SA2_NAME21", "AREASQKM21", "geometry"]],
    how="left",
    predicate="within",
)

if matched["SA2_NAME21"].isna().any():
    print("\nWARNING: Some suburb points did not fall inside an SA2:")
    print(matched[matched["SA2_NAME21"].isna()][["suburb", "suburb_lat", "suburb_lon"]].to_string(index=False))

matched_sa2_names = matched["SA2_NAME21"].dropna().unique().tolist()

sa2_selected = sa2[sa2["SA2_NAME21"].isin(matched_sa2_names)].copy()

# Add suburb labels to selected SA2 boundaries.
sa2_name_to_suburb = matched.dropna(subset=["SA2_NAME21"]).set_index("SA2_NAME21")["suburb"].to_dict()

sa2_selected["suburb"] = sa2_selected["SA2_NAME21"].map(sa2_name_to_suburb)
sa2_selected["sa2_names_used_for_built_up"] = sa2_selected["SA2_NAME21"]

# Project both datasets to EPSG:3857 for intersection areas in m2.
sa2_selected = sa2_selected.to_crs("EPSG:3857")
sa2_selected["sa2_area_m2"] = sa2_selected.geometry.area

layers = fiona.listlayers(BUILT_UP_FILE)
built_layer = "BuiltUpAreas_Source" if "BuiltUpAreas_Source" in layers else layers[0]

built = gpd.read_file(BUILT_UP_FILE, layer=built_layer)
built = built.to_crs("EPSG:3857")
built = built[~built.geometry.isna()].copy()
built = built[built.geometry.is_valid].copy()

if "feature_type" in built.columns:
    built = built[
        built["feature_type"].astype(str).str.lower().str.contains("built", na=False)
    ].copy()

print("\nSA2 proxies selected for built-up area:")
print(
    sa2_selected[
        ["suburb", "SA2_NAME21", "AREASQKM21", "sa2_area_m2"]
    ].sort_values("suburb").to_string(index=False)
)

print("\nIntersecting built-up areas with 12 SA2 proxy boundaries...")

intersection = gpd.overlay(
    built,
    sa2_selected[
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
    raise ValueError("Built-up intersection returned no rows.")

intersection["intersect_area_m2"] = intersection.geometry.area

built_out = (
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

built_out["built_up_area_percent"] = (
    built_out["built_up_area_m2"] / built_out["sa2_area_m2"] * 100
)

built_out["built_up_density_source_note"] = (
    "Digital Atlas Built Up Areas / Bing Building Footprints October 2020; "
    "intersected with ABS SA2 2021 GDA2020 point-selected proxy boundaries. "
    "This represents clustered built-up development area, not literal building footprint coverage."
)

# Keep suburb order.
built_out["suburb"] = pd.Categorical(
    built_out["suburb"],
    categories=suburb_order,
    ordered=True,
)
built_out = built_out.sort_values("suburb").reset_index(drop=True)
built_out["suburb"] = built_out["suburb"].astype(str)

built_out.to_csv(BUILT_OUTPUT, index=False)

print("\nSaved built-up output:")
print(BUILT_OUTPUT)
print(built_out.to_string(index=False))


# =========================
# 5. MERGE ALL INTO STATIC FEATURES
# =========================

print("\n" + "=" * 70)
print("MERGING ALL STATIC FEATURES")
print("=" * 70)

# Remove old versions if rerunning.
drop_cols = [
    "tree_canopy_percent",
    "tree_canopy_area_m2",
    "suburb_area_inside_gsr_m2",
    "cadastred_suburb_area_m2",
    "sa2_name",
    "population_2020",
    "population_density_people_per_km2",
    "area_km2",
    "population_density_source_note",
    "built_up_area_m2",
    "sa2_area_m2",
    "built_up_area_percent",
    "built_up_density_source_note",
    "sa2_names_used_for_built_up",
]

static_clean = static.drop(columns=[c for c in drop_cols if c in static.columns]).copy()

updated = static_clean.merge(tree_out, on="suburb", how="left")
updated = updated.merge(pop_out, on="suburb", how="left")
updated = updated.merge(
    built_out[
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

updated["suburb"] = pd.Categorical(
    updated["suburb"],
    categories=suburb_order,
    ordered=True,
)
updated = updated.sort_values("suburb").reset_index(drop=True)
updated["suburb"] = updated["suburb"].astype(str)

updated.to_csv(STATIC_FILE, index=False)

print("\nUpdated static feature file:")
print(STATIC_FILE)

print("\nFinal static feature table:")
print(updated.to_string(index=False))

print("\nFinal columns:")
print(updated.columns.tolist())

print("\nMissing value check:")
print(updated.isna().sum())

print("\nDone.")
