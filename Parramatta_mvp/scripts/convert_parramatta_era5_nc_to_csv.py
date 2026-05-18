from pathlib import Path

import pandas as pd
import xarray as xr
import numpy as np


# =========================
# 1. PROJECT PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data_raw"
PROCESSED_DIR = BASE_DIR / "data_processed"

RAW_FILE = RAW_DIR / "parramatta_era5_unzipped" / "reanalysis-era5-single-levels-timeseries-sfcfgaepx73.nc"
OUTPUT_FILE = PROCESSED_DIR / "parramatta_era5_hourly_2020.csv"

PROCESSED_DIR.mkdir(exist_ok=True)


# =========================
# 2. CHECK FILE EXISTS
# =========================

if not RAW_FILE.exists():
    raise FileNotFoundError(f"Cannot find ERA5 NetCDF file: {RAW_FILE}")

print("Found ERA5 NetCDF file:")
print(RAW_FILE)


# =========================
# 3. OPEN NETCDF
# =========================

print("\nOpening NetCDF file...")

ds = xr.open_dataset(RAW_FILE)

print("\nDataset:")
print(ds)

print("\nVariables:")
print(list(ds.data_vars))

print("\nCoordinates:")
print(list(ds.coords))


# =========================
# 4. CONVERT TO DATAFRAME
# =========================

df = ds.to_dataframe().reset_index()

print("\nRaw dataframe columns:")
print(df.columns)

print("\nFirst rows:")
print(df.head())


# =========================
# 5. RENAME COMMON ERA5 COLUMNS
# =========================

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

print("\nColumns after renaming:")
print(df.columns)

if "datetime" not in df.columns:
    raise ValueError(
        "Could not find datetime column. "
        f"Available columns are: {list(df.columns)}"
    )


# =========================
# 6. CLEAN TIME
# =========================

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

# ERA5 is UTC. Convert to Sydney local time for night-time cooling analysis.
if df["datetime"].dt.tz is None:
    df["datetime"] = (
        df["datetime"]
        .dt.tz_localize("UTC")
        .dt.tz_convert("Australia/Sydney")
        .dt.tz_localize(None)
    )
else:
    df["datetime"] = (
        df["datetime"]
        .dt.tz_convert("Australia/Sydney")
        .dt.tz_localize(None)
    )


# =========================
# 7. UNIT CONVERSIONS
# =========================

if "temperature_K" in df.columns:
    df["temperature"] = df["temperature_K"] - 273.15
else:
    raise ValueError("Could not find ERA5 temperature column.")

if "dewpoint_K" in df.columns:
    df["dewpoint"] = df["dewpoint_K"] - 273.15

if "wind_u" in df.columns and "wind_v" in df.columns:
    df["wind_speed"] = np.sqrt(df["wind_u"] ** 2 + df["wind_v"] ** 2)

if "air_pressure_Pa" in df.columns:
    df["air_pressure"] = df["air_pressure_Pa"] / 100.0

if "precipitation_m" in df.columns:
    df["precipitation"] = df["precipitation_m"] * 1000.0

# Relative humidity approximation from temperature and dewpoint.
if "temperature" in df.columns and "dewpoint" in df.columns:
    temp = df["temperature"]
    dew = df["dewpoint"]

    saturation_vapour_pressure = 6.112 * np.exp((17.67 * temp) / (temp + 243.5))
    actual_vapour_pressure = 6.112 * np.exp((17.67 * dew) / (dew + 243.5))

    df["humidity"] = 100.0 * actual_vapour_pressure / saturation_vapour_pressure
    df["humidity"] = df["humidity"].clip(lower=0, upper=100)


# =========================
# 8. ADD METADATA
# =========================

df["suburb"] = "Parramatta"
df["suburb_lat"] = -33.815
df["suburb_lon"] = 151.003
df["source"] = "ERA5 hourly time-series"


# =========================
# 9. SELECT FINAL COLUMNS
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

df = df.sort_values("datetime").reset_index(drop=True)


# =========================
# 10. SAVE CSV
# =========================

df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved processed ERA5 CSV:")
print(OUTPUT_FILE)

print("\nPreview:")
print(df.head())
print(df.tail())

print("\nRows:")
print(len(df))

print("\nMissing values:")
print(df.isna().sum())