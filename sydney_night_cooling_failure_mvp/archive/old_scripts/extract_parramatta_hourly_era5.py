from pathlib import Path

import cdsapi
import pandas as pd
import xarray as xr


# =========================
# 1. PROJECT PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data_raw"
PROCESSED_DIR = BASE_DIR / "data_processed"

RAW_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

RAW_FILE = RAW_DIR / "parramatta_era5_hourly_2020.nc"
OUTPUT_FILE = PROCESSED_DIR / "parramatta_era5_hourly_2020.csv"


# =========================
# 2. LOCATION
# =========================

# Parramatta approximate coordinate
LAT = -33.815
LON = 151.003


# =========================
# 3. DOWNLOAD ERA5 HOURLY DATA
# =========================

client = cdsapi.Client()

print("Requesting ERA5 hourly data for Parramatta...")

client.retrieve(
    "reanalysis-era5-single-levels-timeseries",
    {
        "variable": [
            "2m_temperature",
            "2m_dewpoint_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "surface_pressure",
            "total_precipitation",
        ],
        "location": {
            "latitude": LAT,
            "longitude": LON,
        },
        "date": [
            "2020-01-01/2020-12-31",
        ],
        "time": [
            "00:00", "01:00", "02:00", "03:00",
            "04:00", "05:00", "06:00", "07:00",
            "08:00", "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00", "15:00",
            "16:00", "17:00", "18:00", "19:00",
            "20:00", "21:00", "22:00", "23:00",
        ],
        "data_format": "netcdf",
    },
    str(RAW_FILE),
)

print("Downloaded:")
print(RAW_FILE)


# =========================
# 4. LOAD NETCDF
# =========================

ds = xr.open_dataset(RAW_FILE)

print("\nERA5 dataset:")
print(ds)

df = ds.to_dataframe().reset_index()

print("\nRaw dataframe columns:")
print(df.columns)

print("\nFirst rows:")
print(df.head())


# =========================
# 5. STANDARDISE COLUMN NAMES
# =========================

# ERA5 variable names can appear as either short names or long names
# depending on the backend/version. This mapping handles common cases.
rename_map = {
    "valid_time": "datetime",
    "time": "datetime",

    "t2m": "temperature_K",
    "2m_temperature": "temperature_K",

    "d2m": "dewpoint_K",
    "2m_dewpoint_temperature": "dewpoint_K",

    "u10": "wind_u",
    "10m_u_component_of_wind": "wind_u",

    "v10": "wind_v",
    "10m_v_component_of_wind": "wind_v",

    "sp": "air_pressure_Pa",
    "surface_pressure": "air_pressure_Pa",

    "tp": "precipitation_m",
    "total_precipitation": "precipitation_m",
}

df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

if "datetime" not in df.columns:
    raise ValueError(f"Could not find time column. Columns are: {df.columns}")


# =========================
# 6. CLEAN + CONVERT UNITS
# =========================

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

# ERA5 timestamps are UTC. Convert to Sydney local time for night-time analysis.
df["datetime"] = (
    df["datetime"]
    .dt.tz_localize("UTC")
    .dt.tz_convert("Australia/Sydney")
    .dt.tz_localize(None)
)

if "temperature_K" in df.columns:
    df["temperature"] = df["temperature_K"] - 273.15

if "dewpoint_K" in df.columns:
    df["dewpoint"] = df["dewpoint_K"] - 273.15

if "wind_u" in df.columns and "wind_v" in df.columns:
    df["wind_speed"] = (df["wind_u"] ** 2 + df["wind_v"] ** 2) ** 0.5

if "air_pressure_Pa" in df.columns:
    df["air_pressure"] = df["air_pressure_Pa"] / 100.0  # Pa to hPa

if "precipitation_m" in df.columns:
    df["precipitation"] = df["precipitation_m"] * 1000.0  # m to mm

# Approximate relative humidity from temperature and dewpoint.
# This is useful because ERA5 single-level files commonly provide dewpoint,
# not relative humidity directly.
if "temperature" in df.columns and "dewpoint" in df.columns:
    df["humidity"] = 100.0 * (
        (
            6.112 * (2.718281828 ** ((17.67 * df["dewpoint"]) / (df["dewpoint"] + 243.5)))
        )
        /
        (
            6.112 * (2.718281828 ** ((17.67 * df["temperature"]) / (df["temperature"] + 243.5)))
        )
    )

df["suburb"] = "Parramatta"
df["suburb_lat"] = LAT
df["suburb_lon"] = LON
df["source"] = "ERA5 hourly time-series"


# =========================
# 7. SELECT FINAL COLUMNS
# =========================

wanted_cols = [
    "datetime",
    "suburb",
    "suburb_lat",
    "suburb_lon",
    "source",
    "temperature",
    "dewpoint",
    "humidity",
    "wind_speed",
    "air_pressure",
    "precipitation",
]

available_cols = [col for col in wanted_cols if col in df.columns]
df = df[available_cols].copy()

df = df.dropna(subset=["datetime", "temperature"])

df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved processed ERA5 hourly CSV:")
print(OUTPUT_FILE)

print("\nPreview:")
print(df.head())
print(df.tail())

print("\nRows:")
print(len(df))

print("\nMissing values:")
print(df.isna().sum())