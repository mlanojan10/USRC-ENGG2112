from pathlib import Path
import difflib
import pandas as pd


# =========================
# 0. PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

POP_FILE = BASE_DIR / "data_raw" / "population" / "sa2_population_2019_2020.xls"
STATIC_FILE = BASE_DIR / "data_processed" / "suburb_static_features.csv"
POP_OUTPUT = BASE_DIR / "data_processed" / "suburb_population_density_2020_12.csv"
BACKUP_FILE = BASE_DIR / "data_processed" / "suburb_static_features_before_population_12_fix.csv"


# =========================
# 1. CHECK FILES
# =========================

if not POP_FILE.exists():
    raise FileNotFoundError(f"Missing population file: {POP_FILE}")

if not STATIC_FILE.exists():
    raise FileNotFoundError(f"Missing static features file: {STATIC_FILE}")


# =========================
# 2. LOAD ABS POPULATION FILE
# =========================

# From the previous output, header row 6 is correct.
# The problem was using columns 0 and 1 for SA2 code/name.
# In this ABS table, SA2 code/name are columns 8 and 9.
raw = pd.read_excel(POP_FILE, sheet_name="Table 1", header=6)
raw = raw.dropna(axis=1, how="all").copy()

print("Population columns:")
for i, col in enumerate(raw.columns):
    print(i, repr(col))

cols = raw.columns.tolist()

sa2_code_col = cols[8]
sa2_name_col = cols[9]
pop_col = 2020
area_col = "Area"
density_col = "Population density 2020"

required = [sa2_code_col, sa2_name_col, pop_col, area_col, density_col]
missing = [c for c in required if c not in raw.columns]

if missing:
    raise ValueError(f"Missing expected population columns: {missing}")

pop = raw[
    [
        sa2_code_col,
        sa2_name_col,
        pop_col,
        area_col,
        density_col,
    ]
].copy()

pop.columns = [
    "sa2_code",
    "sa2_name",
    "population_2020",
    "area_km2",
    "population_density_people_per_km2",
]

pop["sa2_code"] = pop["sa2_code"].astype(str).str.extract(r"(\d+)")[0]
pop["sa2_name"] = pop["sa2_name"].astype(str).str.strip()

pop["population_2020"] = pd.to_numeric(pop["population_2020"], errors="coerce")
pop["area_km2"] = pd.to_numeric(pop["area_km2"], errors="coerce")
pop["population_density_people_per_km2"] = pd.to_numeric(
    pop["population_density_people_per_km2"],
    errors="coerce",
)

pop = pop.dropna(
    subset=[
        "sa2_code",
        "sa2_name",
        "population_2020",
        "area_km2",
        "population_density_people_per_km2",
    ]
).copy()

print("\nCleaned population table preview:")
print(pop.head(20).to_string(index=False))


# =========================
# 3. CHOOSE SA2 PROXIES
# =========================

# These names are matched against the 2019-20 ABS SA2 population table.
# If one does not match exactly, the script will print candidates and use the closest match.

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

static = pd.read_csv(STATIC_FILE)

suburb_order = static["suburb"].tolist()

print("\nCandidate matches by suburb keyword:")
for suburb in suburb_order:
    candidates = pop[
        pop["sa2_name"].str.contains(suburb, case=False, na=False)
    ][
        [
            "sa2_code",
            "sa2_name",
            "population_2020",
            "area_km2",
            "population_density_people_per_km2",
        ]
    ]

    print("\n" + "=" * 50)
    print(suburb)
    print("=" * 50)
    print(candidates.head(25).to_string(index=False))


# =========================
# 4. EXTRACT POPULATION ROWS
# =========================

rows = []

for suburb in suburb_order:
    target = population_sa2_map[suburb]

    match = pop[pop["sa2_name"] == target]

    if match.empty:
        match = pop[pop["sa2_name"].str.lower() == target.lower()]

    if match.empty:
        close = difflib.get_close_matches(
            target,
            pop["sa2_name"].tolist(),
            n=5,
            cutoff=0.55,
        )

        print(f"\nWARNING: no exact match for {suburb}: {target}")
        print("Close matches:")
        print(close)

        candidates = pop[
            pop["sa2_name"].str.contains(suburb, case=False, na=False)
        ][
            [
                "sa2_code",
                "sa2_name",
                "population_2020",
                "area_km2",
                "population_density_people_per_km2",
            ]
        ]

        print("Candidates containing suburb keyword:")
        print(candidates.head(25).to_string(index=False))

        if close:
            chosen = close[0]
            print(f"Using closest match: {chosen}")
            match = pop[pop["sa2_name"] == chosen]
        elif not candidates.empty:
            chosen = candidates.iloc[0]["sa2_name"]
            print(f"Using first keyword candidate: {chosen}")
            match = pop[pop["sa2_name"] == chosen]
        else:
            rows.append(
                {
                    "suburb": suburb,
                    "sa2_name": target,
                    "population_2020": pd.NA,
                    "population_density_people_per_km2": pd.NA,
                    "area_km2": pd.NA,
                    "population_density_source_note": "ABS Regional Population 2019-20, SA2 proxy; no match found",
                }
            )
            continue

    row = match.iloc[0]

    rows.append(
        {
            "suburb": suburb,
            "sa2_name": row["sa2_name"],
            "population_2020": row["population_2020"],
            "population_density_people_per_km2": row["population_density_people_per_km2"],
            "area_km2": row["area_km2"],
            "population_density_source_note": "ABS Regional Population 2019-20, SA2 proxy",
        }
    )

pop_out = pd.DataFrame(rows)

pop_out["suburb"] = pd.Categorical(
    pop_out["suburb"],
    categories=suburb_order,
    ordered=True,
)
pop_out = pop_out.sort_values("suburb").reset_index(drop=True)
pop_out["suburb"] = pop_out["suburb"].astype(str)

pop_out.to_csv(POP_OUTPUT, index=False)

print("\nSaved population output:")
print(POP_OUTPUT)
print(pop_out.to_string(index=False))


# =========================
# 5. MERGE BACK INTO STATIC FEATURES
# =========================

static.to_csv(BACKUP_FILE, index=False)

drop_cols = [
    "sa2_name",
    "population_2020",
    "population_density_people_per_km2",
    "area_km2",
    "population_density_source_note",
]

static_clean = static.drop(columns=[c for c in drop_cols if c in static.columns]).copy()

updated = static_clean.merge(
    pop_out,
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

print("\nPopulation density check:")
print(
    updated[
        [
            "suburb",
            "sa2_name",
            "population_2020",
            "population_density_people_per_km2",
            "area_km2",
        ]
    ].to_string(index=False)
)

print("\nMissing values:")
print(
    updated[
        [
            "population_2020",
            "population_density_people_per_km2",
            "area_km2",
        ]
    ].isna().sum()
)

print("\nDone.")
