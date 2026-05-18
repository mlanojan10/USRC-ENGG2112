from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data_processed"

STATIC_FILE = PROCESSED_DIR / "suburb_static_features.csv"
CANOPY_FILE = PROCESSED_DIR / "suburb_tree_canopy_2019.csv"
POP_FILE = PROCESSED_DIR / "suburb_population_density_2020.csv"

if not STATIC_FILE.exists():
    raise FileNotFoundError(f"Missing static file: {STATIC_FILE}")

static = pd.read_csv(STATIC_FILE)

print("Original static columns:")
print(static.columns.tolist())

# Keep only the clean base/manual columns.
base_cols = [
    "suburb",
    "distance_to_coast_km",
    "distance_to_water_km",
    "is_coastal",
    "inland_category",
]

missing_base = [c for c in base_cols if c not in static.columns]
if missing_base:
    raise ValueError(f"Missing required base columns in static file: {missing_base}")

clean = static[base_cols].drop_duplicates(subset=["suburb"]).copy()

# Merge canopy cleanly.
if CANOPY_FILE.exists():
    canopy = pd.read_csv(CANOPY_FILE)

    canopy_cols = [
        "suburb",
        "tree_canopy_percent",
        "tree_canopy_area_m2",
        "suburb_area_inside_gsr_m2",
        "cadastred_suburb_area_m2",
    ]

    missing_canopy = [c for c in canopy_cols if c not in canopy.columns]
    if missing_canopy:
        raise ValueError(f"Missing canopy columns: {missing_canopy}")

    clean = clean.merge(
        canopy[canopy_cols],
        on="suburb",
        how="left",
    )
else:
    print(f"Warning: canopy file not found: {CANOPY_FILE}")

# Merge population cleanly.
if POP_FILE.exists():
    pop = pd.read_csv(POP_FILE)

    pop_cols = [
        "suburb",
        "sa2_name",
        "population_2020",
        "population_density_people_per_km2",
        "area_km2",
        "population_density_source_note",
    ]

    missing_pop = [c for c in pop_cols if c not in pop.columns]
    if missing_pop:
        raise ValueError(f"Missing population columns: {missing_pop}")

    clean = clean.merge(
        pop[pop_cols],
        on="suburb",
        how="left",
    )
else:
    print(f"Warning: population file not found: {POP_FILE}")

# Enforce suburb order.
suburb_order = ["Bondi", "Parramatta", "Campbelltown"]
clean["suburb"] = pd.Categorical(clean["suburb"], categories=suburb_order, ordered=True)
clean = clean.sort_values("suburb").reset_index(drop=True)
clean["suburb"] = clean["suburb"].astype(str)

backup = PROCESSED_DIR / "suburb_static_features_messy_backup.csv"
static.to_csv(backup, index=False)

clean.to_csv(STATIC_FILE, index=False)

print("\nSaved clean static feature file:")
print(STATIC_FILE)

print("\nBackup of messy version:")
print(backup)

print("\nClean static features:")
print(clean)

print("\nFinal columns:")
print(clean.columns.tolist())
