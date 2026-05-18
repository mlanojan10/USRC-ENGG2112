from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data_processed"

STATIC_FILE = PROCESSED_DIR / "suburb_static_features.csv"
TREE_FILE = PROCESSED_DIR / "suburb_tree_canopy_2019_12.csv"
POP_FILE = PROCESSED_DIR / "suburb_population_density_2020_12.csv"
BUILT_FILE = PROCESSED_DIR / "suburb_built_up_area_2020_12.csv"

BACKUP_FILE = PROCESSED_DIR / "suburb_static_features_base_backup_before_rebuild.csv"

required_files = [STATIC_FILE, TREE_FILE, POP_FILE, BUILT_FILE]
missing = [str(f) for f in required_files if not f.exists()]

if missing:
    raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

base = pd.read_csv(STATIC_FILE)
tree = pd.read_csv(TREE_FILE)
pop = pd.read_csv(POP_FILE)
built = pd.read_csv(BUILT_FILE)

base.to_csv(BACKUP_FILE, index=False)

# Keep base columns clean.
base_cols = [
    "suburb",
    "distance_to_coast_km",
    "distance_to_water_km",
    "is_coastal",
    "inland_category",
]

base = base[base_cols].copy()

# Remove duplicate population rows if needed.
pop = pop.drop_duplicates(subset=["suburb"]).copy()
tree = tree.drop_duplicates(subset=["suburb"]).copy()
built = built.drop_duplicates(subset=["suburb"]).copy()

# Merge all static features.
updated = base.merge(tree, on="suburb", how="left")
updated = updated.merge(pop, on="suburb", how="left")
updated = updated.merge(
    built[
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

# Fix Bankstown if the population output still has the old bad value.
mask = updated["suburb"] == "Bankstown"
if mask.any():
    updated.loc[mask, "sa2_name"] = "Bankstown - North"
    updated.loc[mask, "population_2020"] = 17736.0
    updated.loc[mask, "area_km2"] = 2.7
    updated.loc[mask, "population_density_people_per_km2"] = 6471.6
    updated.loc[mask, "population_density_source_note"] = (
        "ABS Regional Population 2019-20, SA2 proxy; corrected to Bankstown - North"
    )

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

updated["suburb"] = pd.Categorical(
    updated["suburb"],
    categories=suburb_order,
    ordered=True,
)
updated = updated.sort_values("suburb").reset_index(drop=True)
updated["suburb"] = updated["suburb"].astype(str)

updated.to_csv(STATIC_FILE, index=False)

print("Rebuilt static feature file:")
print(STATIC_FILE)

print("\nKey feature check:")
print(
    updated[
        [
            "suburb",
            "tree_canopy_percent",
            "population_density_people_per_km2",
            "built_up_area_percent",
        ]
    ].to_string(index=False)
)

print("\nMissing values:")
print(
    updated[
        [
            "tree_canopy_percent",
            "population_density_people_per_km2",
            "built_up_area_percent",
        ]
    ].isna().sum()
)

print("\nColumns:")
print(updated.columns.tolist())

print("\nDone.")
