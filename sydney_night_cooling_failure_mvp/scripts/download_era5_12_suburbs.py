from pathlib import Path

import cdsapi
import pandas as pd


# =========================
# 1. PROJECT PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_ERA5_DIR = BASE_DIR / "data_raw" / "era5"
METADATA_DIR = BASE_DIR / "metadata"

RAW_ERA5_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(exist_ok=True)

SUBURB_FILE = METADATA_DIR / "suburbs_12_mvp.csv"


# =========================
# 2. CREATE SUBURB METADATA IF MISSING
# =========================

if not SUBURB_FILE.exists():
    print(f"Creating missing metadata file: {SUBURB_FILE}")

    suburbs_12 = pd.DataFrame(
        [
            ["Bondi", -33.891, 151.276, "coastal", 1, 0.5, 0.5],
            ["Manly", -33.797, 151.287, "coastal", 1, 0.5, 0.5],
            ["Coogee", -33.920, 151.255, "coastal", 1, 0.5, 0.5],
            ["Cronulla", -34.055, 151.152, "coastal", 1, 0.5, 0.5],
            ["Parramatta", -33.815, 151.003, "middle_inland", 0, 20.0, 1.0],
            ["Ryde", -33.815, 151.106, "middle_inland", 0, 13.0, 2.0],
            ["Bankstown", -33.917, 151.034, "middle_inland", 0, 18.0, 5.0],
            ["Fairfield", -33.870, 150.956, "middle_inland", 0, 27.0, 4.0],
            ["Penrith", -33.751, 150.694, "far_inland", 0, 55.0, 3.0],
            ["Blacktown", -33.770, 150.909, "far_inland", 0, 35.0, 6.0],
            ["Liverpool", -33.920, 150.925, "far_inland", 0, 30.0, 2.0],
            ["Campbelltown", -34.067, 150.814, "far_inland", 0, 40.0, 10.0],
        ],
        columns=[
            "suburb",
            "suburb_lat",
            "suburb_lon",
            "inland_category",
            "is_coastal",
            "distance_to_coast_km",
            "distance_to_water_km",
        ],
    )

    suburbs_12["era5_filename"] = (
        suburbs_12["suburb"].str.lower().str.replace(" ", "_", regex=False)
        + "_era5_hourly_2020.nc"
    )

    suburbs_12.to_csv(SUBURB_FILE, index=False)


# =========================
# 3. LOAD SUBURBS
# =========================

suburbs = pd.read_csv(SUBURB_FILE)

required_cols = ["suburb", "suburb_lat", "suburb_lon"]
missing = [c for c in required_cols if c not in suburbs.columns]

if missing:
    raise ValueError(f"Missing columns in {SUBURB_FILE}: {missing}")

print("\nSuburbs to download/check:")
print(suburbs[["suburb", "suburb_lat", "suburb_lon"]])


# =========================
# 4. ERA5 DOWNLOAD SETTINGS
# =========================

client = cdsapi.Client()

variables = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "total_precipitation",
]

hours = [
    "00:00", "01:00", "02:00", "03:00",
    "04:00", "05:00", "06:00", "07:00",
    "08:00", "09:00", "10:00", "11:00",
    "12:00", "13:00", "14:00", "15:00",
    "16:00", "17:00", "18:00", "19:00",
    "20:00", "21:00", "22:00", "23:00",
]


# =========================
# 5. DOWNLOAD ONE FILE PER SUBURB
# =========================

for _, row in suburbs.iterrows():
    suburb = str(row["suburb"])
    lat = float(row["suburb_lat"])
    lon = float(row["suburb_lon"])

    safe_name = suburb.lower().replace(" ", "_")
    out_file = RAW_ERA5_DIR / f"{safe_name}_era5_hourly_2020.nc"

    if out_file.exists() and out_file.stat().st_size > 0:
        print(f"\nSkipping {suburb}, already exists:")
        print(out_file)
        continue

    print("\n" + "=" * 60)
    print(f"Requesting ERA5 hourly data for {suburb}")
    print(f"Latitude: {lat}, Longitude: {lon}")
    print(f"Output: {out_file}")
    print("=" * 60)

    client.retrieve(
        "reanalysis-era5-single-levels-timeseries",
        {
            "variable": variables,
            "location": {
                "latitude": lat,
                "longitude": lon,
            },
            "date": [
                "2020-01-01/2020-12-31",
            ],
            "time": hours,
            "data_format": "netcdf",
        },
        str(out_file),
    )

    print("Downloaded:")
    print(out_file)


# =========================
# 6. FINAL CHECK
# =========================

print("\nDownload/check complete.")

expected_files = []
missing_files = []

for _, row in suburbs.iterrows():
    suburb = str(row["suburb"])
    safe_name = suburb.lower().replace(" ", "_")
    file_path = RAW_ERA5_DIR / f"{safe_name}_era5_hourly_2020.nc"
    expected_files.append(file_path)

    if not file_path.exists() or file_path.stat().st_size == 0:
        missing_files.append(file_path)

print("\nERA5 files now in:")
print(RAW_ERA5_DIR)

print("\nFiles:")
for file_path in expected_files:
    status = "OK" if file_path.exists() and file_path.stat().st_size > 0 else "MISSING"
    size_mb = file_path.stat().st_size / 1_000_000 if file_path.exists() else 0
    print(f"{status:8s} {file_path.name:40s} {size_mb:8.2f} MB")

if missing_files:
    print("\nMissing files:")
    for file_path in missing_files:
        print(file_path)
    raise FileNotFoundError("Some ERA5 files are still missing.")

print("\nAll 12 ERA5 files are present.")
