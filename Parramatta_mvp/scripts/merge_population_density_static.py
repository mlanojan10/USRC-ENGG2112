from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

STATIC_FILE = BASE_DIR / "data_processed" / "suburb_static_features.csv"
POP_FILE = BASE_DIR / "data_processed" / "suburb_population_density_2020.csv"

BACKUP_FILE = BASE_DIR / "data_processed" / "suburb_static_features_before_population.csv"
OUT_FILE = STATIC_FILE

if not STATIC_FILE.exists():
    raise FileNotFoundError(f"Missing static features file: {STATIC_FILE}")

if not POP_FILE.exists():
    raise FileNotFoundError(f"Missing population density file: {POP_FILE}")

static = pd.read_csv(STATIC_FILE)
pop = pd.read_csv(POP_FILE)

static["suburb"] = static["suburb"].astype(str).str.strip().str.title()
pop["suburb"] = pop["suburb"].astype(str).str.strip().str.title()

for col in [
    "population_2020",
    "population_density_people_per_km2",
    "area_km2",
]:
    if col in pop.columns:
        pop[col] = pd.to_numeric(pop[col], errors="coerce")

static.to_csv(BACKUP_FILE, index=False)

drop_cols = [
    "sa2_name",
    "population_2020",
    "population_density_people_per_km2",
    "area_km2",
    "population_density_source_note",
]

static = static.drop(columns=[c for c in drop_cols if c in static.columns])

updated = static.merge(
    pop,
    on="suburb",
    how="left",
)

print(updated)

if updated["population_density_people_per_km2"].isna().any():
    print("\nWARNING: Missing population density for:")
    print(updated[updated["population_density_people_per_km2"].isna()][["suburb"]])

updated.to_csv(OUT_FILE, index=False)

print("\nUpdated static feature file:")
print(OUT_FILE)

print("\nBackup saved:")
print(BACKUP_FILE)
