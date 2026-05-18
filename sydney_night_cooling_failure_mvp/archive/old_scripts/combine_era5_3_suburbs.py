from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# =========================
# 1. PROJECT PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data_raw"
PROCESSED_DIR = BASE_DIR / "data_processed"
METADATA_DIR = BASE_DIR / "metadata"

SUBURB_FILE = METADATA_DIR / "suburbs_3_mvp.csv"
OUTPUT_FILE = PROCESSED_DIR / "sydney_era5_hourly_2020.csv"

PROCESSED_DIR.mkdir(exist_ok=True)


# =========================
# 2. HELPERS
# =========================

def find_col(df, options):
    for option in options:
        if option in df.columns:
            return option
    return None


def calculate_humidity_from_dewpoint(temp_c, dewpoint_c):
    # Magnus approximation.
    return 100.0 * (
        np.exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c))
        /
        np.exp((17.625 * temp_c) / (243.04 + temp_c))
    )


# =========================
# 3. LOAD SUBURB LIST
# =========================

suburbs = pd.read_csv(SUBURB_FILE)

all_rows = []


# =========================
# 4. PROCESS EACH SUBURB FILE
# =========================

for _, row in suburbs.iterrows():
    suburb = str(row["suburb"])
    lat = float(row["suburb_lat"])
    lon = float(row["suburb_lon"])

    safe_name = suburb.lower().replace(" ", "_")
    nc_file = RAW_DIR / f"{safe_name}_era5_hourly_2020.nc"

    if not nc_file.exists():
        raise FileNotFoundError(f"Missing NetCDF file for {suburb}: {nc_file}")

    print(f"\nOpening {suburb}:")
    print(nc_file)

    ds = xr.open_dataset(nc_file)

    print("Variables:", list(ds.data_vars))
    print("Coordinates:", list(ds.coords))

    df = ds.to_dataframe().reset_index()

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
        raise ValueError(f"Could not find datetime column for {suburb}. Columns: {df.columns.tolist()}")

    required_raw = [
        "temperature_K",
        "dewpoint_K",
        "wind_u",
        "wind_v",
        "air_pressure_Pa",
        "precipitation_m",
    ]

    missing_raw = [c for c in required_raw if c not in df.columns]
    if missing_raw:
        raise ValueError(f"Missing required ERA5 columns for {suburb}: {missing_raw}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # ERA5 timestamps are UTC. Convert to Sydney local time for night-time cooling analysis.
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

    df["temperature"] = df["temperature_K"] - 273.15
    df["dewpoint"] = df["dewpoint_K"] - 273.15
    df["humidity"] = calculate_humidity_from_dewpoint(df["temperature"], df["dewpoint"]).clip(0, 100)
    df["wind_speed"] = (df["wind_u"] ** 2 + df["wind_v"] ** 2) ** 0.5
    df["air_pressure"] = df["air_pressure_Pa"] / 100.0
    df["precipitation"] = df["precipitation_m"] * 1000.0

    df["suburb"] = suburb
    df["suburb_lat"] = lat
    df["suburb_lon"] = lon
    df["source"] = "ERA5 hourly time-series"

    final_cols = [
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

    df = df[final_cols].copy()
    df = df.dropna(subset=["datetime", "temperature"])

    # Keep local datetimes that belong to 2020.
    df = df[(df["datetime"] >= "2020-01-01") & (df["datetime"] < "2021-01-01")].copy()

    all_rows.append(df)


# =========================
# 5. COMBINE + SAVE
# =========================

combined = pd.concat(all_rows, ignore_index=True)

combined = combined.sort_values(["suburb", "datetime"]).reset_index(drop=True)

combined.to_csv(OUTPUT_FILE, index=False)

print("\nSaved combined CSV:")
print(OUTPUT_FILE)

print("\nRows per suburb:")
print(combined["suburb"].value_counts())

print("\nFirst rows:")
print(combined.head())

print("\nMissing values:")
print(combined.isna().sum())

print("\nTemperature summary:")
print(combined.groupby("suburb")["temperature"].describe())
