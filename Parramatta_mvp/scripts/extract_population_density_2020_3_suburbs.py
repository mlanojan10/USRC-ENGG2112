from pathlib import Path

import pandas as pd


# =========================
# 0. PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_POP_DIR = BASE_DIR / "data_raw" / "population"
PROCESSED_DIR = BASE_DIR / "data_processed"

OUT_FILE = PROCESSED_DIR / "suburb_population_density_2020.csv"

PROCESSED_DIR.mkdir(exist_ok=True)


# =========================
# 1. FIND ABS FILE
# =========================

files = list(RAW_POP_DIR.glob("*.xls")) + list(RAW_POP_DIR.glob("*.xlsx"))

if not files:
    raise FileNotFoundError(f"No Excel files found in {RAW_POP_DIR}")

POP_FILE = files[0]

print("Using population file:")
print(POP_FILE)


# =========================
# 2. LOAD TABLE 1 = NSW SA2 TABLE
# =========================

raw = pd.read_excel(
    POP_FILE,
    sheet_name="Table 1",
    header=None,
)

print("\nRaw table shape:")
print(raw.shape)

# In this ABS workbook:
# row 7 contains labels like S/T code, S/T name, GCCSA code...
# actual data starts below that.
# Column positions are:
# 8  = SA2 code
# 9  = SA2 name
# 11 = 2020 estimated resident population
# 20 = area km2
# 21 = population density 2020 persons/km2

df = raw.iloc[8:, [8, 9, 11, 20, 21]].copy()

df.columns = [
    "sa2_code",
    "sa2_name",
    "population_2020",
    "area_km2",
    "population_density_people_per_km2",
]

df = df.dropna(subset=["sa2_name"]).copy()

df["sa2_name"] = df["sa2_name"].astype(str).str.strip()

for col in [
    "population_2020",
    "area_km2",
    "population_density_people_per_km2",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nCleaned NSW SA2 population table preview:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())


# =========================
# 3. PRINT CANDIDATES FOR CHECKING
# =========================

search_terms = ["Bondi", "Parramatta", "Campbelltown"]

print("\nCandidate matches:")
for term in search_terms:
    print("\n====================")
    print(term)
    print("====================")

    matches = df[df["sa2_name"].str.contains(term, case=False, na=False)].copy()

    if matches.empty:
        print("No matches found.")
    else:
        print(
            matches[
                [
                    "sa2_code",
                    "sa2_name",
                    "population_2020",
                    "area_km2",
                    "population_density_people_per_km2",
                ]
            ].to_string(index=False)
        )


# =========================
# 4. CHOOSE SA2 PROXIES
# =========================

# These are SA2 proxies for your 3 suburb MVP.
# If one exact name is not found, the script will show candidates
# and you can change the name here.

manual_map = {
    "Bondi": "Bondi Beach - North Bondi",
    "Parramatta": "Parramatta - Rosehill",
    "Campbelltown": "Campbelltown - Woodbine",
}


# =========================
# 5. EXTRACT SELECTED ROWS
# =========================

records = []

for suburb, target_sa2 in manual_map.items():
    exact = df[
        df["sa2_name"].str.lower() == target_sa2.lower()
    ].copy()

    if exact.empty:
        print("\nWARNING: Exact SA2 not found for:")
        print(f"{suburb}: {target_sa2}")

        print("\nAvailable candidates containing suburb name:")
        candidates = df[
            df["sa2_name"].str.contains(suburb, case=False, na=False)
        ].copy()

        if candidates.empty:
            print("No candidates found.")
        else:
            print(
                candidates[
                    [
                        "sa2_code",
                        "sa2_name",
                        "population_2020",
                        "area_km2",
                        "population_density_people_per_km2",
                    ]
                ].to_string(index=False)
            )

        raise ValueError(
            "Edit manual_map in scripts/extract_population_density_2020_3_suburbs.py "
            "using one of the exact SA2 names printed above."
        )

    row = exact.iloc[0]

    records.append(
        {
            "suburb": suburb,
            "sa2_name": row["sa2_name"],
            "population_2020": row["population_2020"],
            "population_density_people_per_km2": row["population_density_people_per_km2"],
            "area_km2": row["area_km2"],
            "population_density_source_note": "ABS Regional Population 2019-20, SA2 proxy",
        }
    )

out = pd.DataFrame(records)

print("\nExtracted 3-suburb population density table:")
print(out)

out.to_csv(OUT_FILE, index=False)

print("\nSaved:")
print(OUT_FILE)
