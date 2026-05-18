from pathlib import Path

import pandas as pd
import xarray as xr


# =========================
# 0. PROJECT PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data_raw"
PROCESSED_DIR = BASE_DIR / "data_processed"

PROCESSED_DIR.mkdir(exist_ok=True)

INPUT_FILE = RAW_DIR / "parramatta_wind_2020_2039_raw.nc"
OUTPUT_FILE = PROCESSED_DIR / "parramatta_wind_2020.csv"


# =========================
# 1. CHECK INPUT FILE
# =========================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Missing file: {INPUT_FILE}\n"
        "Place parramatta_wind_2020_2039_raw.nc inside data_raw/"
    )


# =========================
# 2. LOAD NETCDF
# =========================

ds = xr.open_dataset(INPUT_FILE)

print("NetCDF dataset:")
print(ds)

print("\nAvailable variables:")
print(list(ds.data_vars))

print("\nAvailable coordinates:")
print(list(ds.coords))


# =========================
# 3. DETECT VARIABLE AND COORDINATES
# =========================

wind_var = list(ds.data_vars)[0]

if "lat" in ds.coords:
    lat_name = "lat"
elif "latitude" in ds.coords:
    lat_name = "latitude"
else:
    raise ValueError("Could not find latitude coordinate.")

if "lon" in ds.coords:
    lon_name = "lon"
elif "longitude" in ds.coords:
    lon_name = "longitude"
else:
    raise ValueError("Could not find longitude coordinate.")

if "time" in ds.coords:
    time_name = "time"
else:
    raise ValueError("Could not find time coordinate.")

print("\nUsing wind variable:", wind_var)
print("Using latitude coordinate:", lat_name)
print("Using longitude coordinate:", lon_name)
print("Using time coordinate:", time_name)


# =========================
# 4. CHECK FILE IS NOT EMPTY
# =========================

if ds.sizes.get(lat_name, 0) == 0 or ds.sizes.get(lon_name, 0) == 0:
    raise ValueError(
        "Downloaded wind NetCDF has zero lat/lon grid cells. "
        "Redownload using a larger Parramatta bounding box."
    )


# =========================
# 5. EXTRACT NEAREST PARRAMATTA GRID CELL
# =========================

parramatta_lat = -33.82
parramatta_lon = 151.00

wind_point = ds[wind_var].sel(
    {
        lat_name: parramatta_lat,
        lon_name: parramatta_lon,
    },
    method="nearest",
)

print("\nSelected nearest grid cell:")
print(wind_point)


# =========================
# 6. CONVERT TO DATAFRAME
# =========================

wind_df = wind_point.to_dataframe().reset_index()

print("\nRaw extracted dataframe:")
print(wind_df.head())
print(wind_df.columns)


# =========================
# 7. CLEAN DATAFRAME
# =========================

wind_df = wind_df.rename(
    columns={
        time_name: "date",
        wind_var: "wind_speed_mean",
    }
)

wind_df = wind_df[["date", "wind_speed_mean"]].copy()

wind_df["date"] = pd.to_datetime(wind_df["date"], errors="coerce")
wind_df["wind_speed_mean"] = pd.to_numeric(
    wind_df["wind_speed_mean"],
    errors="coerce",
)

wind_df = wind_df.dropna(subset=["date", "wind_speed_mean"])


# =========================
# 8. FILTER ONLY 2020
# =========================

wind_df = wind_df[
    (wind_df["date"] >= "2020-01-01")
    & (wind_df["date"] < "2021-01-01")
].copy()


# =========================
# 9. CONVERT TO DAILY MEAN
# =========================

wind_df = (
    wind_df
    .groupby(wind_df["date"].dt.date)["wind_speed_mean"]
    .mean()
    .reset_index()
)

wind_df["date"] = pd.to_datetime(wind_df["date"])


# =========================
# 10. SAVE CLEAN CSV
# =========================

wind_df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved clean wind file:")
print(OUTPUT_FILE)

print("\nFirst 5 rows:")
print(wind_df.head())

print("\nLast 5 rows:")
print(wind_df.tail())

print("\nRows saved:")
print(len(wind_df))