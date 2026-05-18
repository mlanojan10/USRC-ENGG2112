from pathlib import Path
from datetime import datetime
import os
import ssl

import pandas as pd


# =========================
# 0. SSL FALLBACK
# =========================

# Meteostat downloads public weather CSV files over HTTPS.
# On some macOS Python installs, certificate verification can fail.
# This disables SSL verification only for this script.
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["PYTHONHTTPSVERIFY"] = "0"


# Import meteostat AFTER SSL fallback
import meteostat as ms


# =========================
# 1. PROJECT PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data_processed"
PROCESSED_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "parramatta_hourly_2020.csv"


# =========================
# 2. LOCATION + DATE RANGE
# =========================

# Parramatta approximate coordinates
parramatta = ms.Point(-33.82, 151.00, 20)

start = datetime(2020, 1, 1, 0, 0)
end = datetime(2020, 12, 31, 23, 59)


# =========================
# 3. FIND NEARBY STATIONS
# =========================

print("Finding nearby Meteostat stations...")

stations_df = ms.stations.nearby(parramatta, limit=20)

if stations_df is None or stations_df.empty:
    raise ValueError("No nearby Meteostat stations found for Parramatta.")

print("\nNearby Meteostat stations:")
print(stations_df)

print("\nStation IDs:")
print(stations_df.index)


# =========================
# 4. FETCH HOURLY DATA
# =========================

print("\nFetching Meteostat hourly data...")

df = None
selected_station_id = None
selected_station_name = None

for station_id, row in stations_df.iterrows():
    station_name = row.get("name", "Unknown station")

    print(f"\nTrying station {station_id}: {station_name}")

    try:
        station = ms.Station(id=station_id)
        data = ms.hourly(station, start, end)
        temp_df = data.fetch()
    except Exception as e:
        print(f"Failed for station {station_id}: {e}")
        continue

    if temp_df is not None and not temp_df.empty:
        df = temp_df
        selected_station_id = station_id
        selected_station_name = station_name
        print(f"Success. Using station {station_id}: {station_name}")
        break

    print(f"No data returned for station {station_id}.")

if df is None:
    raise ValueError(
        "Meteostat returned no usable hourly data from nearby stations. "
        "This may be a network issue, station coverage issue, or date-range issue."
    )

print("\nRaw Meteostat hourly data:")
print(df.head())
print(df.tail())

print("\nSelected station:")
print(selected_station_id, selected_station_name)

print("\nColumns:")
print(df.columns)

print("\nRows:")
print(len(df))


# =========================
# 5. CLEAN DATA
# =========================

df = df.reset_index()

df = df.rename(
    columns={
        "time": "datetime",
        "temp": "temperature",
        "rhum": "humidity",
        "wspd": "wind_speed",
        "pres": "air_pressure",
        "prcp": "precipitation",
    }
)

wanted_cols = [
    "datetime",
    "temperature",
    "humidity",
    "wind_speed",
    "air_pressure",
    "precipitation",
]

available_cols = [col for col in wanted_cols if col in df.columns]

df = df[available_cols].copy()

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

for col in df.columns:
    if col != "datetime":
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["datetime", "temperature"])

df["suburb"] = "Parramatta"
df["station_id"] = selected_station_id
df["station_name"] = selected_station_name

df = df[
    ["datetime", "suburb", "station_id", "station_name"]
    + [
        col
        for col in df.columns
        if col not in ["datetime", "suburb", "station_id", "station_name"]
    ]
]


# =========================
# 6. SAVE
# =========================

df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved hourly file:")
print(OUTPUT_FILE)

print("\nCleaned hourly data:")
print(df.head())
print(df.tail())

print("\nRows saved:")
print(len(df))

print("\nMissing values:")
print(df.isna().sum())