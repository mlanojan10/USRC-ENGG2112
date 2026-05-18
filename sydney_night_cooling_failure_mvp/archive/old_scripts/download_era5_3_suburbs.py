from pathlib import Path

import cdsapi
import pandas as pd


# =========================
# 1. PROJECT PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data_raw"
METADATA_DIR = BASE_DIR / "metadata"

RAW_DIR.mkdir(exist_ok=True)
METADATA_DIR.mkdir(exist_ok=True)

SUBURB_FILE = METADATA_DIR / "suburbs_3_mvp.csv"


# =========================
# 2. LOAD SUBURBS
# =========================

suburbs = pd.read_csv(SUBURB_FILE)

required_cols = ["suburb", "suburb_lat", "suburb_lon"]
missing = [c for c in required_cols if c not in suburbs.columns]

if missing:
    raise ValueError(f"Missing columns in {SUBURB_FILE}: {missing}")


# =========================
# 3. ERA5 DOWNLOAD SETTINGS
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
# 4. DOWNLOAD ONE FILE PER SUBURB
# =========================

for _, row in suburbs.iterrows():
    suburb = str(row["suburb"])
    lat = float(row["suburb_lat"])
    lon = float(row["suburb_lon"])

    safe_name = suburb.lower().replace(" ", "_")
    out_file = RAW_DIR / f"{safe_name}_era5_hourly_2020.nc"

    if out_file.exists():
        print(f"Skipping {suburb}, already exists:")
        print(out_file)
        continue

    print(f"\nRequesting ERA5 hourly data for {suburb}...")
    print(f"Latitude: {lat}, Longitude: {lon}")

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


print("\nAll requested downloads complete.")
